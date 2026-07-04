"""Triton tile renderer - linear space only. Verified against manual_renderer.py."""
import torch, triton, triton.language as tl
from torch.autograd import Function

FILL=0.9; D2R=3.141592653589793/180.0

def _aabbs(cx,cy,rx,ry,ang,pad=3.0):
    r=ang*D2R;ca=torch.abs(torch.cos(r));sa=torch.abs(torch.sin(r))
    return torch.stack([cx-(rx*ca+ry*sa+pad),cy-(rx*sa+ry*ca+pad),
                        cx+(rx*ca+ry*sa+pad),cy+(rx*sa+ry*ca+pad)],dim=-1)

def _tiles(aabbs,H,W,ts=128):
    ny=(H+ts-1)//ts;nx=(W+ts-1)//ts
    x0,y0,x1,y1=aabbs[:,0],aabbs[:,1],aabbs[:,2],aabbs[:,3]
    parts,offs=[],[0]
    for ty in range(ny):
        for tx in range(nx):
            X0,Y0=tx*ts,ty*ts;X1,Y1=min(X0+ts,W),min(Y0+ts,H)
            m=(x1>X0)&(x0<X1)&(y1>Y0)&(y0<Y1)
            idx=torch.where(m)[0].to(torch.int32)
            parts.append(idx);offs.append(offs[-1]+len(idx))
    return(torch.cat(parts)if parts else torch.zeros(0,dtype=torch.int32),
           torch.tensor(offs,dtype=torch.int32),nx,ny)

@triton.jit
def _fwd(TMPL,TIDX,CX,CY,RX,RY,ANG,COL,OP,TOFF,TSHAPES,OUT,
         T:tl.constexpr,H:tl.constexpr,W:tl.constexpr,TS:tl.constexpr,NTX:tl.constexpr,
         BGR:tl.constexpr,BGG:tl.constexpr,BGB:tl.constexpr,FL:tl.constexpr,DR:tl.constexpr,
         PXB:tl.constexpr):
    pid=tl.program_id(0);ty=pid//NTX;tx=pid%NTX
    tx0=tx*TS;ty0=ty*TS;tx1=tl.minimum(tx0+TS,W);ty1=tl.minimum(ty0+TS,H)
    th=ty1-ty0;tw=tx1-tx0;st=tl.load(TOFF+pid);ed=tl.load(TOFF+pid+1);ns=ed-st;HT=0.5*(T-1)
    for ps in range(0,th*tw,PXB):
        off=ps+tl.arange(0,PXB);mk=off<th*tw
        py=ty0+(off//tw);px=tx0+(off%tw)
        Cr=tl.full([PXB],BGR,tl.float32);Cg=tl.full([PXB],BGG,tl.float32)
        Cb=tl.full([PXB],BGB,tl.float32);Tt=tl.full([PXB],1.0,tl.float32)
        for s in range(ns):
            si=tl.load(TSHAPES+st+s)
            scx=tl.load(CX+si);scy=tl.load(CY+si);srx=tl.load(RX+si)+1e-8;sry=tl.load(RY+si)+1e-8
            sang=tl.load(ANG+si)*DR;sop=tl.load(OP+si);tidx=tl.load(TIDX+si)
            dx=px.to(tl.float32)-scx;dy=py.to(tl.float32)-scy
            ca=tl.cos(-sang);sa=tl.sin(-sang);dxr=dx*ca-dy*sa;dyr=dx*sa+dy*ca
            tmpx=dxr/srx*FL;tmpy=dyr/sry*FL;u=(tmpx+1.0)*HT;v=(tmpy+1.0)*HT
            u=tl.clamp(u,0.0,T-1.001);v=tl.clamp(v,0.0,T-1.001)
            x0=u.to(tl.int32);y0=v.to(tl.int32);x1=tl.minimum(x0+1,T-1);y1=tl.minimum(y0+1,T-1)
            fx=u-x0.to(tl.float32);fy=v-y0.to(tl.float32)
            bp=TMPL+tidx*T*T
            v00=tl.load(bp+y0*T+x0,mask=mk,other=0.0);v10=tl.load(bp+y0*T+x1,mask=mk,other=0.0)
            v01=tl.load(bp+y1*T+x0,mask=mk,other=0.0);v11=tl.load(bp+y1*T+x1,mask=mk,other=0.0)
            ar=(1-fx)*(1-fy)*v00+fx*(1-fy)*v10+(1-fx)*fy*v01+fx*fy*v11
            a=tl.where(ar>0.5,1.0,0.0)*sop;a=tl.clamp(a,0.0,1.0)
            cr=tl.load(COL+si*3+0);cg=tl.load(COL+si*3+1);cb=tl.load(COL+si*3+2)
            w=a*Tt;Cr+=w*cr;Cg+=w*cg;Cb+=w*cb;Tt*=(1.0-a)
        i3=(py*W+px)*3
        tl.store(OUT+i3+0,Cr,mask=mk);tl.store(OUT+i3+1,Cg,mask=mk);tl.store(OUT+i3+2,Cb,mask=mk)

@triton.jit
def _bwd(STMPL,TIDX,CX,CY,RX,RY,ANG,COL,OP,TOFF,TSHAPES,GO,GCX,GCY,GRX,GRY,GANG,GCOL,GOP,
         TSCRATCH,  # [ntiles * MAX_NS] scratch for T values
         T:tl.constexpr,H:tl.constexpr,W:tl.constexpr,TS:tl.constexpr,NTX:tl.constexpr,
         FL:tl.constexpr,DR:tl.constexpr,MAX_NS:tl.constexpr,PXB:tl.constexpr):
    pid=tl.program_id(0);ty=pid//NTX;tx=pid%NTX
    tx0=tx*TS;ty0=ty*TS;tx1=tl.minimum(tx0+TS,W);ty1=tl.minimum(ty0+TS,H)
    th=ty1-ty0;tw=tx1-tx0;st=tl.load(TOFF+pid);ed=tl.load(TOFF+pid+1);ns=ed-st;HT=0.5*(T-1)
    for ps in range(0,th*tw,PXB):
        off=ps+tl.arange(0,PXB);mk=off<th*tw
        py=ty0+(off//tw);px=tx0+(off%tw);i3=(py*W+px)*3
        gor=tl.load(GO+i3+0,mask=mk,other=0.0);gog=tl.load(GO+i3+1,mask=mk,other=0.0)
        gob=tl.load(GO+i3+2,mask=mk,other=0.0)
        # Phase 1: recompute soft forward, store T_before in TSCRATCH
        Cr=tl.full([PXB],0.0,tl.float32);Cg=tl.full([PXB],0.0,tl.float32)
        Cb=tl.full([PXB],0.0,tl.float32);Tf=tl.full([PXB],1.0,tl.float32)
        for s in range(ns):
            # Store T_before: one value per pixel, PXB values at consecutive addrs
            tbase = TSCRATCH + (pid * MAX_NS + s) * PXB
            tl.store(tbase + tl.arange(0, PXB), Tf, mask=mk)
            si=tl.load(TSHAPES+st+s)
            scx=tl.load(CX+si);scy=tl.load(CY+si);srx=tl.load(RX+si)+1e-8;sry=tl.load(RY+si)+1e-8
            sang=tl.load(ANG+si)*DR;sop=tl.load(OP+si);tidx=tl.load(TIDX+si)
            dx=px.to(tl.float32)-scx;dy=py.to(tl.float32)-scy
            ca=tl.cos(-sang);sa=tl.sin(-sang);dxr=dx*ca-dy*sa;dyr=dx*sa+dy*ca
            tmpx=dxr/srx*FL;tmpy=dyr/sry*FL;u=(tmpx+1.0)*HT;v=(tmpy+1.0)*HT
            u=tl.clamp(u,0.0,T-1.001);v=tl.clamp(v,0.0,T-1.001)
            x0=u.to(tl.int32);y0=v.to(tl.int32);x1=tl.minimum(x0+1,T-1);y1=tl.minimum(y0+1,T-1)
            fx=u-x0.to(tl.float32);fy=v-y0.to(tl.float32)
            bp=STMPL+tidx*T*T
            v00=tl.load(bp+y0*T+x0,mask=mk,other=0.0);v10=tl.load(bp+y0*T+x1,mask=mk,other=0.0)
            v01=tl.load(bp+y1*T+x0,mask=mk,other=0.0);v11=tl.load(bp+y1*T+x1,mask=mk,other=0.0)
            ar=(1-fx)*(1-fy)*v00+fx*(1-fy)*v10+(1-fx)*fy*v01+fx*fy*v11
            a=tl.clamp(ar*sop,0.0,1.0)
            cr=tl.load(COL+si*3+0);cg=tl.load(COL+si*3+1);cb=tl.load(COL+si*3+2)
            w=a*Tf;Cr+=w*cr;Cg+=w*cg;Cb+=w*cb;Tf*=(1-a)
        dCr=gor;dCg=gog;dCb=gob;dT=tl.full([PXB],0.0,tl.float32)
        for s in range(ns-1,-1,-1):
            si=tl.load(TSHAPES+st+s)
            scx=tl.load(CX+si);scy=tl.load(CY+si);srx=tl.load(RX+si)+1e-8;sry=tl.load(RY+si)+1e-8
            sang=tl.load(ANG+si)*DR;sop=tl.load(OP+si);tidx=tl.load(TIDX+si)
            dx=px.to(tl.float32)-scx;dy=py.to(tl.float32)-scy
            ca=tl.cos(-sang);sa=tl.sin(-sang);dxr=dx*ca-dy*sa;dyr=dx*sa+dy*ca
            tmpx=dxr/srx*FL;tmpy=dyr/sry*FL;u=(tmpx+1.0)*HT;v=(tmpy+1.0)*HT
            u=tl.clamp(u,0.0,T-1.001);v=tl.clamp(v,0.0,T-1.001)
            x0=u.to(tl.int32);y0=v.to(tl.int32);x1=tl.minimum(x0+1,T-1);y1=tl.minimum(y0+1,T-1)
            fx=u-x0.to(tl.float32);fy=v-y0.to(tl.float32)
            bp=STMPL+tidx*T*T
            v00=tl.load(bp+y0*T+x0,mask=mk,other=0.0);v10=tl.load(bp+y0*T+x1,mask=mk,other=0.0)
            v01=tl.load(bp+y1*T+x0,mask=mk,other=0.0);v11=tl.load(bp+y1*T+x1,mask=mk,other=0.0)
            ar=(1-fx)*(1-fy)*v00+fx*(1-fy)*v10+(1-fx)*fy*v01+fx*fy*v11
            a=tl.clamp(ar*sop,0.0,1.0)
            da_du=(1-fy)*(v10-v00)+fy*(v11-v01);da_dv=(1-fx)*(v01-v00)+fx*(v11-v10)
            cr=tl.load(COL+si*3+0);cg=tl.load(COL+si*3+1);cb=tl.load(COL+si*3+2)
            Tp=tl.load(TSCRATCH + (pid*MAX_NS + s)*PXB + tl.arange(0,PXB), mask=mk, other=1.0)
            dot=dCr*cr+dCg*cg+dCb*cb;dLda=(dot-dT)*Tp
            tl.atomic_add(GOP+si,tl.sum(dLda*ar,axis=0))
            tl.atomic_add(GCOL+si*3+0,tl.sum(dCr*Tp*a,axis=0))
            tl.atomic_add(GCOL+si*3+1,tl.sum(dCg*Tp*a,axis=0))
            tl.atomic_add(GCOL+si*3+2,tl.sum(dCb*Tp*a,axis=0))
            dLdtx=dLda*da_du*HT*sop;dLdty=dLda*da_dv*HT*sop
            irx=1.0/srx;iry=1.0/sry
            dLdcx=dLdtx*(-ca*irx*FL)+dLdty*(-sa*iry*FL)
            dLdcy=dLdtx*(sa*irx*FL)+dLdty*(-ca*iry*FL)
            dLdrx=dLdtx*(-tmpx*irx);dLdry=dLdty*(-tmpy*iry)
            dLdphi=dLdtx*(-dyr*irx*FL)+dLdty*(dxr*iry*FL);dLdang=dLdphi*(-DR)
            tl.atomic_add(GCX+si,tl.sum(dLdcx,axis=0));tl.atomic_add(GCY+si,tl.sum(dLdcy,axis=0))
            tl.atomic_add(GRX+si,tl.sum(dLdrx,axis=0));tl.atomic_add(GRY+si,tl.sum(dLdry,axis=0))
            tl.atomic_add(GANG+si,tl.sum(dLdang,axis=0))
            dT=dT*(1-a)+dot*a

def _ens(x,dt=torch.float32):return x.contiguous().to(dt)

def triton_fwd(htmpl,tidx,cx,cy,rx,ry,ang,col_lin,op,H,W,bg_lin):
    dev=htmpl.device;Tsz=htmpl.shape[1]
    htmpl=_ens(htmpl);tidx=_ens(tidx,torch.int32);cx=_ens(cx);cy=_ens(cy)
    rx=_ens(rx);ry=_ens(ry);ang=_ens(ang);col_lin=_ens(col_lin);op=_ens(op);bg_lin=_ens(bg_lin)
    ab=_aabbs(cx,cy,rx,ry,ang);ts,to,ntx,nty=_tiles(ab,H,W)
    ts=_ens(ts,torch.int32).to(dev);to=_ens(to,torch.int32).to(dev);nt=to.shape[0]-1
    out=torch.empty(H,W,3,device=dev,dtype=torch.float32)
    bgr,bgg,bgb=bg_lin[0].item(),bg_lin[1].item(),bg_lin[2].item()
    _fwd[(nt,)](htmpl,tidx,cx,cy,rx,ry,ang,col_lin,op,to,ts,out,
                T=Tsz,H=H,W=W,TS=128,NTX=ntx,BGR=bgr,BGG=bgg,BGB=bgb,FL=FILL,DR=D2R,PXB=256)
    return out

def triton_bwd(stmpl,tidx,cx,cy,rx,ry,ang,col_lin,op,go_lin,H,W):
    dev=stmpl.device;N=cx.shape[0];Tsz=stmpl.shape[1]
    if go_lin.dim()==3 and go_lin.shape[0]==3:go_lin=go_lin.permute(1,2,0)
    stmpl=_ens(stmpl);tidx=_ens(tidx,torch.int32);cx=_ens(cx);cy=_ens(cy)
    rx=_ens(rx);ry=_ens(ry);ang=_ens(ang);col_lin=_ens(col_lin);op=_ens(op);go_lin=_ens(go_lin)
    ab=_aabbs(cx,cy,rx,ry,ang);ts,to,ntx,nty=_tiles(ab,H,W)
    ts=_ens(ts,torch.int32).to(dev);to=_ens(to,torch.int32).to(dev);nt=to.shape[0]-1
    gc=torch.zeros(N,device=dev);gy=torch.zeros(N,device=dev)
    gx=torch.zeros(N,device=dev);gr=torch.zeros(N,device=dev)
    ga=torch.zeros(N,device=dev);gcol=torch.zeros(N,3,device=dev);gop=torch.zeros(N,device=dev)
    # Scratch: nt tiles × MAX_NS shapes × PXB pixels (256)
    MAX_NS=512; PXB=256
    tscratch=torch.zeros(nt,MAX_NS,PXB,device=dev,dtype=torch.float32)
    _bwd[(nt,)](stmpl,tidx,cx,cy,rx,ry,ang,col_lin,op,to,ts,go_lin,
                gc,gy,gx,gr,ga,gcol,gop,tscratch,
                T=Tsz,H=H,W=W,TS=128,NTX=ntx,FL=FILL,DR=D2R,MAX_NS=MAX_NS,PXB=PXB)
    return gc,gy,gx,gr,ga,gcol,gop

class TritonV2STE(Function):
    @staticmethod
    def forward(ctx,h,s,ti,cx,cy,rx,ry,ang,col,op,H,W,bg_lin):
        ctx.save_for_backward(s,ti,cx.detach(),cy.detach(),rx.detach(),ry.detach(),
                              ang.detach(),col.detach(),op.detach())
        ctx.H,ctx.W=H,W
        with torch.no_grad():return triton_fwd(h,ti,cx,cy,rx,ry,ang,col,op,H,W,bg_lin)
    @staticmethod
    def backward(ctx,go):
        s,ti,cx,cy,rx,ry,ang,col,op=ctx.saved_tensors;H,W=ctx.H,ctx.W
        g=triton_bwd(s,ti,cx,cy,rx,ry,ang,col,op,go,H,W)
        return(None,None,None,g[0],g[1],g[2],g[3],g[4],g[5],g[6],None,None,None)

def check_triton_vs_tiled(ns=20,nt=4,H=128,W=128,dev='cuda',ts=32):
    from .templates import generate_synthetic_templates
    from .manual_renderer import over_composite_forward_tiled,over_composite_backward_tiled
    from .loss import srgb_to_linear, linear_to_srgb
    from .manual_renderer import _srgb_to_linear_grad, _linear_to_srgb_grad
    lib=generate_synthetic_templates(num_types=nt,device=dev)
    torch.manual_seed(42)
    cx=torch.rand(ns,device=dev)*W;cy=torch.rand(ns,device=dev)*H
    rx=torch.rand(ns,device=dev)*20+5;ry=torch.rand(ns,device=dev)*20+5
    ang=torch.rand(ns,device=dev)*360;col=torch.rand(ns,3,device=dev)
    op=torch.rand(ns,device=dev)*0.5+0.5;ti=torch.randint(0,nt,(ns,),device=dev)
    bg=torch.zeros(3,device=dev);bg_lin=srgb_to_linear(bg)
    # Forward: PyTorch (sRGB) vs Triton (linear鈫抯RGB)
    with torch.no_grad():
        opy=over_composite_forward_tiled(lib['hard'],ti,cx,cy,rx,ry,ang,col,op,H,W,bg,dev,ts)
        col_lin=srgb_to_linear(col)
        otr_lin=triton_fwd(lib['hard'],ti,cx,cy,rx,ry,ang,col_lin,op,H,W,bg_lin)
        otr=linear_to_srgb(otr_lin.permute(2,0,1))
    fd=(opy-otr).abs().max().item()
    # Backward: convert dL/d(srgb) 鈫?dL/d(linear) for Triton
    go_srgb=torch.randn(3,H,W,device=dev)
    gpy=over_composite_backward_tiled(lib['soft'],ti,cx,cy,rx,ry,ang,col,op,H,W,go_srgb,dev,ts)
    # Compute dL/d(linear) = dL/d(srgb) * d(srgb)/d(linear)
    with torch.no_grad():
        C_lin=otr_lin.permute(2,0,1)
    dsrgb_dlin=_linear_to_srgb_grad(C_lin)
    go_lin=go_srgb*dsrgb_dlin
    gtr_raw=triton_bwd(lib['soft'],ti,cx,cy,rx,ry,ang,col_lin,op,go_lin,H,W)
    # Convert color gradients back: dL/d(srgb_color) = dL/d(lin_color)*d(lin)/d(srgb)
    dlin_dsrgb=_srgb_to_linear_grad(col)
    gtr_col=gtr_raw[5]*dlin_dsrgb
    gd={'cx':gtr_raw[0],'cy':gtr_raw[1],'rx':gtr_raw[2],'ry':gtr_raw[3],
        'angle':gtr_raw[4],'colors':gtr_col,'opacity':gtr_raw[6]}
    r={'fwd_max_diff':fd};ok=fd<1e-3
    for n in['cx','cy','rx','ry','angle','colors','opacity']:
        d=(gpy[n]-gd[n]).abs().max().item();m=gpy[n].abs().max().item()
        rel=d/(m+1e-8);o=rel<0.05 or d<1e-3
        r[n]={'diff':d,'rel':rel,'ok':o}
        if not o:ok=False
    r['_all_ok']=ok;return r
