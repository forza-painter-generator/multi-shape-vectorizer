# FH6 Multi-Shape Differentiable Vectorizer — 可行性分析与实现方案

> **目标**: 结合 vinylizer 的可微渲染优化管线、diffbmp 的多形状方案、IGS 的 Triton 高速 kernel、forza-painter-fh6 的图形数据与导入管道，实现一个 **Python/PyTorch + Triton** 项目，能用 FH6 游戏中数十种涂装图形高质量还原任意原图的自动化工具。
>
> **目标语言**: Python 3.10+ / PyTorch 2.x / Triton / CUDA (通过 `torch.utils.cpp_extension` 或 Triton kernel)

---

## 目录

1. [项目概述与背景](#1-项目概述与背景)
2. [源码关键位置索引](#2-源码关键位置索引)
3. [核心技术原理分析](#3-核心技术原理分析)
4. [损失函数优化方案](#4-损失函数优化方案)
5. [IGS Triton Kernel 分析与可复用性](#5-igs-triton-kernel-分析与可复用性)
6. [FH6 游戏图形系统](#6-fh6-游戏图形系统)
7. [可行性分析](#7-可行性分析)
8. [实现方案](#8-实现方案)
9. [图形精选策略](#9-图形精选策略)
10. [潜在风险与缓解](#10-潜在风险与缓解)

---

## 1. 项目概述与背景

### 1.1 四个关键项目

| 项目 | 仓库 | 论文 | 作用 | 语言（与新项目的关系） |
|------|------|------|------|------|
| **vinylizer** | `e:\workspace\vinylizer` | 无（个人项目） | 可微渲染 + 梯度下降 + 循环重定位算法（**参考算法逻辑**） | C++/CUDA |
| **diffbmp** | https://github.com/smhongok/diffbmp | https://arxiv.org/abs/2602.22625 (CVPR 2026) | 多形状可微渲染框架（**架构可直接复用**） | Python/PyTorch |
| **IGS** | https://github.com/KohakuBlueleaf/IGS | 无 | Triton 2D Gaussian Splatting 高速 kernel（**kernel 可直接改造复用**） | Python/Triton |
| **forza-painter-fh6** | `e:\workspace\forza-painter-fh6` | 无 | FH6 图形数据 + JSON 导入/导出（**导入代码可直接复用**） | Python |

### 1.2 当前限制与机遇

- **vinylizer**: 只能用 1 种图形（渐变椭圆 type 228）。C++ 实现，算法逻辑（循环重定位、STE、Adam 融合 kernel）可参考但需移植到 Python/PyTorch。
- **diffbmp**: 支持任意 bitmap 图元，使用 Gaussian blur 解决硬边缘不可导问题。**Python/PyTorch 实现，渲染+优化框架可直接复用**。但不了解 FH6 内存布局和图形系统。
- **IGS**: 拥有高性能 Triton kernel（内核融合 + Chunking），**Python/Triton，kernel 结构可直接改造复用**。但渲染模型是归一化加权和而非 Over 合成，需改造。
- **forza-painter-fh6**: 拥有完整的 FH6 图形目录（~1000+ 种）、导入管道。**Python 实现，导入代码可直接复用**。

**合成这四个项目，用 Python/PyTorch + Triton 实现。Python 生态带来关键优势**：diffbmp 渲染框架直接可用（不必从 C++ 移植）、IGS Triton kernel 直接可用（不必手写 CUDA）、PyTorch autograd 自动处理链式法则（不必手推导数）、VGG perceptual loss 开箱即用（不必 OpenCV DNN 绕路）、AI 辅助编程 token 消耗低（Python 比 C++ 简洁 3-5×）。

---

## 2. 源码关键位置索引

### 2.1 vinylizer (`e:\workspace\vinylizer`)

| 内容 | 文件 | 关键行/说明 |
|------|------|------------|
| **alpha_228 软椭圆公式** | `src/common/alpha_228.h` | 完整文件 — 隐式曲线 + 牛顿法 + LUT 预计算 |
| **Shape/Layer 数据结构** | `src/common/types.h` | 第 15-70 行 — Shape 类型定义，`SHAPE_RECTANGLE=1`, `ROTATED_ELLIPSE=16`, `SOFT_ELLIPSE_228=228` |
| **前向渲染 kernel (含硬椭圆)** | `src/cuda/canvas_render.cu` | 第 28-88 行 `canvas_render_kernel` — 三分支：矩形/硬椭圆/软椭圆 over 合成 |
| **优化器前向渲染 kernel** | `src/cuda/render_kernel.cu` | 第 95-145 行 `render_tile_forward_kernel` — 仅使用 alpha_228，保存 scratch 用于反向 |
| **反向传播 kernel** | `src/cuda/render_kernel.cu` | 第 148-300 行 `render_backward_kernel` — 完整链式法则，alpha→rn→dsq→几何参数 |
| **MSE 损失 + 梯度** | `src/core/diff_renderer.cu` | 第 143-190 行 `mse_grad_kernel` — RGB+Alpha 双通道损失 |
| **Adam 更新 + 参数 clamp + STE** | `src/core/diff_renderer.cu` | 第 248-320 行 `fused_adam_kernel` |
| **DiffRenderer 接口** | `src/core/diff_renderer.h` | 完整文件 — GPU 缓冲区结构与 CUDA 函数声明 |
| **主优化循环** | `src/core/optimizer.cpp` | 第 260-510 行 `gradient_optimize()` — Cycle 循环：Phase A 全局优化 → Phase B 扫描无用图形 → Phase C 局部优化 |
| **重定位逻辑** | `src/core/optimizer.cpp` | 第 510-680 行 `find_relocation_candidates()` + `relocate_shapes()` |
| **Pipeline 入口** | `src/core/pipeline.h` + `src/core/pipeline.cpp` | PipelineConfig 结构 + `run_pipeline()` |
| **最终渲染 Canvas** | `src/common/canvas.h` + `src/common/canvas.cpp` | 用于生成预览 PNG |
| **JSON 输出** | `src/output/json_writer.cpp` | Shapes → JSON 序列化 |
| **FH6 内存注入** | `src/inject/inject.cpp` + `src/inject/inject.h` | 直接写入 FH6 进程内存 |
| **颜色工具** | `src/cuda/color_utils.cuh` | sRGB ↔ Linear 转换 |
| **预处理** | `src/preprocess/preprocessor.h` + `.cpp` | K-means 颜色量化、边缘检测、显著性检测 |

### 2.2 diffbmp (`E:\workspace\diffbmp`)

| 内容 | 文件 | 说明 |
|------|------|------|
| **论文** | https://arxiv.org/abs/2602.22625 | CVPR 2026, "soft rasterization via Gaussian blur" |
| **GitHub** | https://github.com/smhongok/diffbmp | 完整源码 |
| **Gaussian Blur 工具** | `pydiffbmp/util/utils.py` | `gaussian_blur()` — 可分离 2D 高斯模糊 |
| **Primitive 加载 + 模糊** | `pydiffbmp/util/primitive_loader.py` | 加载 SVG/PNG/JPG 为 bitmap |
| **VectorRenderer 初始化 (blur 应用处)** | `pydiffbmp/core/renderer/vector_renderer.py` | 第 70-95 行 — `self.S_blurred = gaussian_blur(self.S, sigma)` |
| **CUDA 前向 + 双线性采样** | `cuda_tile_rasterizer/cuda_kernels/tile_forward.cu` | `render_tile_forward_kernel` — 坐标变换 → `bilinear_sample` 模糊模板 |
| **CUDA 反向 + 梯度** | `cuda_tile_rasterizer/cuda_kernels/tile_backward.cu` | `backward_over_one_pixel` — 通过 `bilinear_value_and_grad_xy` 回传梯度 |
| **Tile 渲染器 (Python 侧)** | `pydiffbmp/core/renderer/simple_tile_renderer.py` | `render_from_params()` — sigma/blur 管道入口 |
| **优化主循环** | `main.py` | `do_gaussian_blur` 配置 → 控制 sigma |
| **配置示例** | `configs/default.json` | `do_gaussian_blur: true`, `c_blend`, loss 配置 |
| **软光栅化可视化** | `pydiffbmp/util/visualize_gaussian_blur.py` | `batched_soft_rasterize()` — 展示 blur 效果 |
| **Loss 函数注册表** | `pydiffbmp/util/loss_functions.py` | 第 1-380 行 — MSE/L1/Huber/Perceptual/SSIM/Edge/Alpha/Grayscale 全套 |
| **Perceptual Loss 实现** | `pydiffbmp/util/loss_functions.py` | 第 85-130 行 — VGG 特征提取 + 特征空间 MSE |
| **SSIM/Edge 失败记录** | `pydiffbmp/util/loss_functions.py` | 第 155-180 行 — 明确标注不可用及失败原因 |
| **Loss 组合器** | `pydiffbmp/util/loss_functions.py` | `LossComposer` 类 — 加权组合多 loss |
| **推荐 loss 权重** | `configs/default.json` | 第 33-38 行 — `mse:1.0 + perceptual:0.2` |

### 2.3 forza-painter-fh6 (`E:\workspace\forza-painter-fh6`)

| 内容 | 文件 | 说明 |
|------|------|------|
| **图形家族基础码** | `src/fh6_vinyl_resources.py` | `VINYL_TYPE_BASES` — 28 个图形家族的 type_code 基数 |
| **图形目录 CSV** | `src/data/FH6 Shape Library Data - FH6 Shape Library Data.csv` | 完整 type_code 目录 |
| **图形几何数据** | `src/data/fh6_vinyl_resources/Vinyls/<Family>/<index>` | JSON 文件，含 Vertices + Indices |
| **图形预览 PNG** | `src/data/fh6_vinyl_resources/Vinyls/<Family>/<index>.png` | 图形预览图 |
| **Shape 类型定义** | `src/geometry_json.py` | `ShapeType` enum: `RECTANGLE=1`, `ROTATED_ELLIPSE=16` |
| **导出格式参考** | `src/fh6_typecode_export.py` | 从 FH6 内存读取并导出 JSON |
| **导入逻辑** | `src/fh6_typecode_import.py` | 解析 JSON → 写入 FH6 内存 |
| **内存层结构** | `src/fh6_typecode_import.py` | 第 108-135 行 `decode()` — 关键偏移量 |
| **字体注册表** | `src/data/fh6_font_registry.json` | 字体字形 → shape_word 映射 |
| **类型码常量** | `src/fh6_shape_catalog.py` | `TYPECODE_BASE = 0x100000`，图形解析 |

### 2.4 IGS (`E:\workspace\IGS`)

| 内容 | 文件 | 说明 |
|------|------|------|
| **GitHub** | https://github.com/KohakuBlueleaf/IGS | 2D Gaussian Splatting |
| **朴素实现 (参考)** | `src/igs/gs2d.py` | `naive_gaussian_2d()` — 归一化加权和公式 |
| **Triton 融合前向 kernel** | `src/igs/gs_triton.py` | 第 20-170 行 — fused weight+matmul, 无中间张量 |
| **Triton 融合反向 kernel** | `src/igs/gs_triton.py` | 第 250-440 行 — Recompute forward in backward |
| **Triton Chunked 前向 kernel** | `src/igs/gs_triton_chunked.py` | 第 25-210 行 — N_CHUNK 并行 + `tl.dot` 矩阵乘法 |
| **Chunked 反向 kernel (有性能问题)** | `src/igs/gs_triton_chunked.py` | 第 220+ 行 — 标注 "THIS ONE have performance issue" |
| **自动调优 block size** | `src/igs/gs_triton.py` | 第 445-448 行 — `BLOCK_SIZE_H/W` 参数 |
| **核心差异说明** | `README.md` | 归一化加权和 vs Over 合成 |

---

## 3. 核心技术原理分析

### 3.1 可微分渲染的整体架构

三个项目共享同一范式：

```
┌─────────────────────────────────────────────────────────┐
│                   可微渲染优化管线                         │
│                                                         │
│  1. 前向 Pass: 参数 → CUDA Over 合成 → 渲染图            │
│  2. 损失计算: MSE(渲染图, 原图) + 可选感知损失             │
│  3. 反向 Pass: 梯度链式传播 → dL/d(每个参数)              │
│  4. 参数更新: Adam 优化器 + cosine annealing LR          │
│  5. 重定位: 回收"无用"图元，放到误差最大区域               │
│  6. 重复 1-5 直到收敛                                    │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Over 合成（前向渲染）

所有项目都使用 Porter-Duff over 操作，在 GPU 上逐像素从后往前合成：

```
for each pixel p:
    C = (0,0,0), T = 1.0
    for each shape from back to front:
        α_eff = alpha(p) × opacity
        w = α_eff × T
        C += w × srgb_to_linear(color)
        T *= (1 - α_eff)
        if T < ε: break   // 早期终止优化
    C += T × background
    output = linear_to_srgb(C)
```

颜色混合在**线性空间**进行（`srgb_to_linear`/`linear_to_srgb`），确保物理正确的颜色合成。

### 3.3 两种"软化"策略对比

这是理解两个项目差异的核心：

#### vinylizer: 数学定义的软函数

```cpp
// src/common/alpha_228.h — 隐式曲线
// 1 - C1*a - C2*a² - C3*a³ = r_n
// 牛顿迭代解出 a = alpha(r_n)
// 在 r_n∈[0,1] 区间内 alpha 从 1.0 平滑降到 0.0

// 反向求导 (render_kernel.cu):
float denom = C1_228 + 2*C2_228*a + 3*C3_228*a*a;
float dadr = -1.0f / denom;  // da/dr_n
float daddsq = dadr / (2 * rn);  // da/d(r_n²)
// 然后链式传导到 cx,cy,rx,ry,angle
```

- **优点**: 精确、无额外存储、原生 CUDA 性能最优
- **缺点**: 只支持一种形状（只设计了一个数学公式）

#### diffbmp: Gaussian Blur 预处理

```python
# pydiffbmp/core/renderer/vector_renderer.py
if sigma > 0.0:
    bmp = self.S.unsqueeze(0)      # 原始硬 bitmap {0,1}
    bmp = gaussian_blur(bmp, sigma) # 高斯模糊 → 连续值 [0,1]
    self.S_blurred = bmp
```

```cuda
// CUDA kernel: 前向用双线性插值采样模糊模板
mask_value = bilinear_sample(template_blurred, tex_x, tex_y);
// mask_value 现在是连续的 [0,1]，不再是二值！
```

```cuda
// 反向: bilinear_sample 在 PyTorch/CUDA 中原生可微
// bilinear_value_and_grad_xy() 返回 mask 值 + 空间梯度
mask_val, dmask_dx_tex, dmask_dy_tex = bilinear_value_and_grad_xy(...)
// 梯度通过 F.grid_sample 的 autograd 自动回传
```

- **优点**: 支持任意 bitmap 形状
- **缺点**: σ=1.0 在 128px 模板上 ≈ 3px 过渡带，在 2000px 画布上 ≈ 0.5-1px（基本不可见）

### 3.4 为什么硬边缘不可导

硬椭圆的 alpha 函数是阶跃函数：`α = H(1 - r_n²)`。

- $dα/d(r_n²) = -δ(1 - r_n²)$（Dirac δ 函数）
- 椭圆内部梯度 = 0（不知道"该变大还是变小"）
- 椭圆外部梯度 = 0（不知道"该往哪移动"）
- 边界处梯度 = ∞（数值不稳定）

**解决方法就是两种"软化"策略之一。**

### 3.5 循环重定位 (Cyclic Relocation)

vinylizer 的核心创新（`src/core/optimizer.cpp` 第 260-510 行）：

```
每个大周期 = Phase A (全局 Adam 150步) + Phase B (扫描) + Phase C (局部 50步)

Phase B 扫描逻辑:
  1. 跟踪每个 shape 最近 50 步的梯度范数
  2. 梯度持续很小的 = "稳定收敛"
  3. 其中 blend weight 很低的 = "无用"（太小或被盖住）
  4. 无用图形 → 重新初始化到当前误差最大的区域
  5. 新图形追加到 z-order 最顶层

Phase C:
  - 冻结旧图形（frozen_mask）
  - 只优化被重定位的新图形 50 步
  - 然后清除 frozen_mask 进行下一轮全局优化
```

### 3.6 Straight-Through Estimator (STE)

vinylizer 在颜色量化上已经使用了 STE：

```cuda
// 前向: 量化到整数（游戏需要 8-bit）
ste_colors[i] = roundf(float_colors[i]);

// 反向: 梯度不受量化影响（直接从未量化的参数求导）
// 这是 STE 的核心思想
```

同样的思路可以扩展到 shape type：前向用硬图形渲染（匹配游戏），反向用软图形求梯度。

---

## 5. 损失函数优化方案

> **核心观点**: 纯 MSE loss 只关心逐像素 RGB 差值，对人类的视觉感知不是最优。引入 perceptual loss 可以显著提升视觉质量。

### 4.1 MSE 的局限性

MSE 把每个像素的 RGB 误差平等对待：

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i} (C_i^{\text{rendered}} - C_i^{\text{target}})^2$$

问题：
- **对微小偏移不敏感**: 渲染图比原图偏移 1px，MSE 很大但人眼觉得"基本一样"
- **对结构信息无感知**: 把像素打乱但颜色分布不变，MSE 不变但人眼觉得天差地别
- **高频纹理惩罚过重**: 迫使优化器把有限图元浪费在噪点上

### 4.2 Perceptual Loss (VGG-based)

**原理**: 不比较像素，而是比较 VGG 网络中间层的特征图。特征图捕获的是纹理、结构、语义信息，而非精确颜色。

```python
# 参考: diffbmp/pydiffbmp/util/loss_functions.py 第 85-130 行
class LossRegistry:
    @staticmethod
    def perceptual_loss(rendered, target, vgg_model):
        # 1. 图像归一化到 ImageNet 统计量
        rendered_norm = (rendered - mean) / std
        target_norm = (target - mean) / std
        
        # 2. 提取 VGG 中间层特征 (通常是 relu3_3 或 relu4_3)
        with torch.no_grad():
            target_features = vgg_model(target_norm)
        rendered_features = vgg_model(rendered_norm)
        
        # 3. 在特征空间算 MSE
        loss = F.mse_loss(rendered_features, target_features) / 100.0
        return loss
```

**为什么 VGG 特征更好**:
- VGG 的卷积核天然对边缘、纹理、形状敏感
- 浅层特征 (relu2_2) 捕获边缘和颜色斑点
- 中层特征 (relu3_3) 捕获纹理和图样
- 深层特征 (relu4_3) 捕获语义结构

**对本项目的意义**: 用 3000 个硬图形拟合照片时，perceptual loss 会让优化器优先还原"人眼关心"的结构（轮廓、面部特征、文字），而不是逐像素拟合噪点。

### 4.3 推荐组合方案

参考 diffbmp 的验证结论（`configs/default.json` 第 33-38 行）：

```json
{
    "loss_config": {
        "type": "combined",
        "components": [
            {"name": "mse",        "weight": 1.0},
            {"name": "perceptual", "weight": 0.2}
        ]
    }
}
```

即：

$$\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{\text{MSE}} + 0.2 \cdot \mathcal{L}_{\text{perceptual}}$$

**权重设计理由**: MSE 主导（确保颜色准确），perceptual 辅助（引导结构正确）。perceptual weight 不宜过大（0.1-0.3），否则会过度平滑导致细节丢失。

### 4.4 各 Loss 函数对比

| Loss | 当前状态 | 适用场景 | 原因 |
|------|---------|---------|------|
| **MSE** | ✅ vinylizer 已使用 | 所有场景 | 基础像素级保真 |
| **Perceptual (VGG)** | ✅ diffbmp 已验证可用 | **推荐添加** | 提升结构感知质量 |
| **L1 / Huber** | 未使用 | 可替代 MSE | 对 outlier 更鲁棒，颜色过渡更平滑 |
| **SSIM** | ❌ diffbmp 已验证不可用 | — | 与梯度下降冲突，loss 不降反升 (+21.7%) |
| **Edge Loss (Sobel)** | ❌ diffbmp 已验证不可用 | — | 梯度与其他 loss 冲突，loss 不降反升 |
| **Grayscale MSE** | 未使用 | 可选 | 权宜方案：降低颜色敏感度，关注亮度结构 |

> **来源**: diffbmp `loss_functions.py` 第 155-180 行明确标注 SSIM 和 Edge loss 为 `NotImplementedError`，附带失败原因说明。

### 4.5 在 Python 项目中实现 Perceptual Loss（开箱即用）

由于项目是 Python/PyTorch，Perceptual Loss **直接可用，零开发成本**：

```python
import torch
import torchvision.models as models
import torch.nn.functional as F

# 1. 加载预训练 VGG16（只取特征提取层）
class VGGFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice = torch.nn.Sequential(*[vgg[i] for i in range(16)])  # relu3_3
        for p in self.parameters():
            p.requires_grad = False  # VGG 不参与训练
    
    def forward(self, x):
        return self.slice(x)

vgg = VGGFeatureExtractor().to(device).eval()

# 2. 每步计算组合损失
def compute_loss(rendered, target):
    # MSE: 像素级保真
    mse = F.mse_loss(rendered, target)
    
    # Perceptual: VGG 特征空间相似度
    with torch.no_grad():
        feat_target = vgg(preprocess_imagenet(target))
    feat_rendered = vgg(preprocess_imagenet(rendered))
    perceptual = F.mse_loss(feat_rendered, feat_target) / 100.0
    
    return mse + 0.2 * perceptual  # 权重参考 diffbmp
```

**不需要** OpenCV DNN、不需要手写 CUDA backward、不需要 ONNX 模型转换。`torchvision` 直接提供预训练权重。

> **参考**: diffbmp `loss_functions.py` 第 85-130 行的 `perceptual_loss` 实现，以及 `configs/default.json` 第 33-38 行的权重配置 `mse:1.0 + perceptual:0.2`。

### 4.6 推荐的渐进实现路径 (Python/PyTorch 项目)

| 阶段 | 内容 | 复杂度 | 收益 |
|------|------|--------|------|
| **Phase 1** | 纯 MSE + `torch.optim.Adam`，先跑通多形状版本 | 🟢 基准 | 基准 |
| **Phase 2** | 添加 Grayscale MSE 作为可选项 | 🟢 ~10行 | 亮度结构敏感度 ↑ |
| **Phase 3** | 添加 VGG Perceptual Loss（`torchvision` 直接加载） | 🟢 **~20行** | ⭐ 视觉质量明显提升 |
| **Phase 4 (可选)** | 尝试 LPIPS 等更高级的感知损失 | 🟡 ~30行 | 可能进一步提升 |

> **Phase 3 在 Python 下仅需 ~20 行**（vs C++ 版需要 OpenCV DNN 等 ~200 行）。`torchvision.models.vgg16(weights='IMAGENET1K_V1')` 一行搞定。

### 4.7 L1/Huber Loss（极低成本，Python 版）

PyTorch 原生支持，无需手写 CUDA kernel：

```python
# L1 loss
l1 = F.l1_loss(rendered, target)

# Huber loss (smooth L1)
huber = F.smooth_l1_loss(rendered, target, beta=1.0)

# 组合: mse + perceptual + l1
total = 1.0 * F.mse_loss(...) + 0.2 * perceptual_loss(...) + 0.1 * F.l1_loss(...)
```

L1 对 outlier 更鲁棒：某个图元颜色完全错了，不会产生巨大的梯度把其他参数带偏。

---

## 6. IGS Triton 高速 Kernel 分析与可复用性

> **来源**: [IGS (Image Gaussian Splatting)](https://github.com/KohakuBlueleaf/IGS) — `E:\workspace\IGS`

### 5.1 IGS 是什么

IGS 是一个用 **2D Gaussian Splatting** 做图像拟合的项目。和 vinylizer/diffbmp 核心差异：

| | vinylizer | diffbmp | IGS |
|---|----------|---------|-----|
| **图元类型** | 渐变椭圆 (alpha_228) | 任意 bitmap | 2D Gaussian 核 |
| **alpha 计算** | 隐式曲线 + LUT | bilinear_sample(模板) | `exp(-0.5 × 马氏距离)` |
| **合成方式** | Over (alpha compositing) | Over | **归一化加权和** (不同于 Over!) |
| **GPU 实现** | 手写 CUDA | 手写 CUDA + PyTorch | **Triton** (更高级) |
| **关键优化** | tile AABB 过滤 | tile AABB 过滤 | **内核融合 + Chunking** |

### 5.2 关键发现：IGS 的合成模型与我们不同

这是最重要的区别。IGS 用的是 **normalized weighted sum**：

```python
# IGS: 归一化加权和
weight = exp(-0.5 * distance) * alpha
result = sum(weight_i * color_i) / (sum(weight_i) + eps)  # 除以所有权重之和

# vinylizer/diffbmp: Over compositing (Porter-Duff)
for i from back to front:
    C += T * alpha_i * color_i
    T *= (1 - alpha_i)
```

**归一化加权和的含义**：所有 Gaussian 的贡献除以总权重。这意味着后画的 Gaussian 不一定覆盖前面的，顺序无关。

**Over 合成的含义**：z-order 重要，先画的被后画的遮挡。这是 FH6 的渲染方式。

**结论**：IGS 的渲染模型和 FH6 不兼容。如果用户期望"在游戏里看起来一样"，必须使用 Over 合成。所以 **IGS 的渲染公式不能直接用**。

### 5.3 但 IGS 的优化技术有部分可迁移

#### 5.3.1 内核融合 (Kernel Fusion) — ⚠️ 部分可用

IGS 的核心创新：将 weight 计算 + matmul 融合进一个 kernel，**完全避免 [B, N, H, W] 的中间张量**。

```python
# Naive 方法 (diffbmp/vinylizer 当前做法):
# 步骤1: 前向 → weight[B,N,H,W]  (写入全局显存)
# 步骤2: weight_sum = weight.sum(dim=1)
# 步骤3: 反向 → 读 weight back

# IGS 融合方法:
# 前向: 直接累加 atomicAdd(result, weight*color) 和 atomicAdd(weight_sum, weight)
# 反向: 重新计算 forward (Recompute) + 直接累加梯度
# → 完全不需要存储 weight[B,N,H,W]!
```

**但在我们的场景中**：vinylizer 已经用了 scratch buffer 来避免存 weight（`alpha_scratch` + `r_norm_scratch`，只存 per-tile per-shape，不是 full H×W）。diffbmp 的 tile renderer 也同样只在 tile 级别存 scratch。所以 **这个优化的增量收益不如预期大**。

#### 5.3.2 Chunked Gaussian Processing — ✅ 非常有用

```python
# IGS gs_triton_chunked.py: 将 N 个 Gaussian 分成多个 chunk 处理
N_CHUNK_SIZE = 32  # 每个 chunk 处理 32 个 Gaussian
for chunk in range(0, N, N_CHUNK_SIZE):
    # 一次性加载 chunk 的所有参数到寄存器
    pos_x = tl.load(positions[chunk:chunk+N_CHUNK_SIZE])
    # 在 chunk 内用 tl.dot 做高效的矩阵乘法
    result = tl.dot(weight_reshaped, colors_chunk)
```

**核心优势**: 减少了 atomicAdd 冲突。多个 thread block 同时 atomicAdd 到同一个像素会串行化。Chunking 让一次 kernel launch 内处理多个 Gaussian，block 内部先做归约再 atomicAdd，冲突减少 $N_\text{chunk}$ 倍。

**如何迁移到我们的 pipeline**:

```cuda
// 当前 vinylizer: 每个 shape 独立一个 kernel
// render_backward_kernel: 逐 shape 处理，atomicAdd 冲突多

// 改造为 chunked:
// 每个 thread block 处理 CHUNK_SIZE(=16) 个 shape
// block 内部用 shared memory 归约梯度
// 然后只做一次 atomicAdd
__global__ void render_backward_chunked_kernel(
    ..., int CHUNK_SIZE) {
    __shared__ float s_grad_cx[16][THREADS_PER_BLOCK];
    // 对 chunk 内每个 shape:
    for (int ci = 0; ci < CHUNK_SIZE; ++ci) {
        int i = chunk_start + ci;
        // ... 计算每个像素对 shape_i 的梯度贡献 ...
        // 累加到 shared memory
    }
    // chunk 处理完，归约 shared memory
    // 然后只做一次 atomicAdd
}
```

**对 3000 个 shape 的效果**: 当前每个 backward 迭代 ~3000 个 kernel launch。Chunked 版本降至 ~3000/16 ≈ 188 个 kernel launch。Kernel launch overhead 在 CUDA 中约 5-10μs，3000 次就是 15-30ms/迭代。

#### 5.3.3 Recompute Forward in Backward — ✅ 已经部分采用

IGS 的反向 kernel 不依赖 forward 保存的中间结果，而是**重新计算 forward 值**：

```python
# IGS backward: 重新计算而不是读取缓存
dx = x_block - pos_x
dy = y_block - pos_y
distance = cov_inv_00*(dx*dx) + cov_inv_11*(dy*dy) + cov_inv_01*(2*dx*dy)
gaussian_val = tl.exp(-0.5 * distance)
```

vinylizer 已经部分采用了这个策略——它用 `alpha_scratch` 和 `r_norm_scratch` 保存了中间值，但 scratch 本身也要占用显存。对于 bitmap 模板来说，recompute 的成本更高（需要重新做 bilinear_sample），但可以省去每 shape 每 tile 的 scratch 显存。

**判断**: 如果模板数量少（~20 种），scratch 显存不大。如果扩展到 100+ 种模板，recompute 策略更优。

#### 5.3.4 Triton — ✅ Python 项目下强烈推荐

由于目标项目是 Python/PyTorch，Triton 是**最优 GPU kernel 方案**：

**Triton 的核心优势（对 Python 项目）**:
- 自动调优 block size（`@triton.autotune`），无需手工搜索最优参数
- 语法接近 Python，AI 辅助编程非常友好
- 跨 GPU 架构兼容（Ampere → Hopper → Blackwell 无需改代码）
- 在 PyTorch 生态中原生集成（`torch.autograd.Function`），梯度流无缝对接

**最关键的好处**: IGS 的 `gs_triton.py` 和 `gs_triton_chunked.py` 提供了完整的 fused forward+backward Triton kernel 实现。我们只需要：
1. 将 Gaussian 距离公式（`exp(-0.5*distance)`）→ 改为 bilinear_sample 模板采样
2. 将归一化加权和（`sum/(sum_weight+eps)`）→ 改为 Over compositing（`T *= (1-alpha)` 累乘）
3. 其余结构（chunking、shared memory 归约、recompute forward）全部复用

**迁移成本对比**:
| 方案 | 手写 CUDA C++ | 改造 IGS Triton |
|------|-------------|----------------|
| 代码量 | ~500 行 CUDA C++ + pybind | **~150 行 Python** |
| 调试难度 | 高（cuda-gdb） | **低（Python + Triton debug mode）** |
| 自动调优 | 需手工 block size | **`@triton.autotune`** |
| 与 PyTorch 集成 | 需手写 pybind | **原生 `torch.autograd.Function`** |

### 5.4 实际可复用的优化清单（Python 项目视角）

| 优化 | 来源 | 可复用性 | 预期收益 | 实现成本 |
|------|------|---------|---------|---------|
| **Triton 改造 IGS kernel** | `gs_triton_chunked.py` | ✅ **强烈推荐** | Kernel 融合 + Chunking + 自动调优 | ~150行 Python |
| **Chunked primitive processing** | `gs_triton_chunked.py` | ✅ 高 | Kernel launch 减少 ~16× | Triton 原生支持 |
| **Chunk 内 shared memory 归约** | `gs_triton_chunked.py:160-195` | ✅ 高 | atomicAdd 冲突减少 | Triton 原生支持 |
| **Recompute forward in backward** | `gs_triton.py:250-400` | ⚠️ 有条件 | 省 scratch 显存 | 模板多时有用 |
| **PyTorch autograd** | `torch.autograd` | ✅ 零成本 | 替代手写链式法则 | 0 行（自带） |
| **`torch.optim.Adam`** | `torch.optim` | ✅ 零成本 | 替代手写 CUDA Adam | 0 行（自带） |
| **归一化加权和** | `naive_gaussian_2d` | ❌ 不兼容 FH6 | 渲染模型不同 | — |

### 5.5 推荐：立即采用的优化

**Chunked primitive processing** 是性价比最高的优化，可以这样改造 vinylizer 的 `render_backward_kernel`：

```cuda
// 改造思路: 将 tile forward+backward 改为 per-chunk 版本
// 当前: per-tile loop { filter active shapes → per-shape kernel }
// 改造: per-chunk loop { load N_CHUNK shapes → per-chunk kernel (内部归约) }

#define N_CHUNK 16  // 每个 chunk 处理 16 个 shape

__global__ void tile_forward_backward_chunked_kernel(
    ..., int chunk_start, int chunk_size) {
    
    // 1. 加载 chunk 内所有 shape 的参数到寄存器/shared memory
    __shared__ float s_cx[N_CHUNK], s_cy[N_CHUNK];
    __shared__ float s_rx[N_CHUNK], s_ry[N_CHUNK];
    // ... 从 global memory 加载
    
    // 2. 逐像素遍历 tile
    // 3. Chunk 内的 shape 在 shared memory 中归约梯度
    // 4. 最后只做一次 atomicAdd 到 global grad buffer
    
    for (int ci = 0; ci < chunk_size; ++ci) {
        // ... per-pixel per-shape 梯度计算 ...
        // 累加到 shared memory 梯度累加器
        s_grad_cx[ci][threadIdx.x] += local_grad_cx;
    }
    
    // Warp-level 归约 + 仅一次 atomicAdd
    if (threadIdx.x == 0) {
        for (int ci = 0; ci < chunk_size; ++ci) {
            atomicAdd(&grad_cx[chunk_start + ci], s_grad_cx[ci][0]);
        }
    }
}
```

### 5.6 不推荐的优化及其原因

1. **归一化加权和替代 Over**: ❌ FH6 的渲染模型是 z-order Over compositing，换成归一化加权和在游戏里看起来完全不同。
2. **完全去掉 scratch buffer 改用 recompute**: ⚠️ 对于 20 种模板，scratch 显存仅 ~50MB（可接受）。如果扩展 100+ 种模板再考虑。
3. **手写 CUDA C++ kernel 替代 Triton**: ❌ 在 Python 项目中，Triton 的开发效率远超手写 CUDA + pybind。Triton 自动调优能力在跨 GPU 部署时尤其有价值。

---

## 6. FH6 游戏图形系统

### 6.1 Type Code 编码

```
type_code = 0x100000 + shape_word

shape_word 由家族 base + index 决定:
  Primitives:         base = 1048677
  Gradient_Shapes:    base = 1048777
  Stripes:            base = 1048877
  ... (共 28 个家族)
```

### 6.2 图形几何数据格式

```json
// 示例: src/data/fh6_vinyl_resources/Vinyls/Primitives/2 (圆形)
{
  "Info": {"Type": 1048677, "TypeIndex": 2},
  "Vertices": [
    {"X": 0.0, "Y": -64.25},
    {"X": -16.63, "Y": -62.06},
    ... // 49 个三角形顶点
  ],
  "Indices": [0, 1, 2, 3, 0, 4, ...],  // 三角形索引
  "VerticesAlpha": "////////..."        // base64 顶点透明度
}
```

### 6.3 FH6 内存层结构（每层 0x140 字节）

```
Offset  Size   Field
0x18    8      position (float x, float y)
0x28    8      scale (float sx, float sy)
0x50    4      rotation (float, 度)
0x74    4      color (uint8 RGBA)
0x7A    2      shape_word (uint16)
               → type_code = 0x100000 + shape_word
```

### 6.4 导入 JSON 格式

```json
{
  "shapes": [
    {
      "type": 1048677,
      "data": [cx, -cy,       // FH6 的 Y 轴是翻转的
               sx, sy,         // scale
               rotation,       // 角度
               0],             // skew
      "color": [R, G, B, A],
      "mask": 0
    }
  ]
}
```

---

## 7. 可行性分析

### 7.1 三项目功能矩阵

```
                      vinylizer    diffbmp     forza-painter   新项目
                      ────────     ──────      ────────────   ──────
FH6 内存注入           ✅            ❌           ✅             ✅ (复用)
FH6 图层结构           ✅            ❌           ✅             ✅ (复用)
GPU Adam 优化          ✅            ❌           ❌             ✅ (改造)
循环重定位              ✅            ❌           ❌             ✅ (保留)
多形状可微渲染          ❌            ✅           ❌             ✅ (新增)
Gaussian blur 软化      ❌            ✅           ❌             ✅ (新增)
STE 硬/软分离           ❌            ❌           ❌             ✅ (新增)
1000+ 图形几何数据      ❌            ❌           ✅             ✅ (复用)
导入 JSON 生成         ✅ (只椭圆)    ❌           ✅             ✅ (改造)
```

### 7.2 关键挑战与方案

| 挑战 | 方案 | 风险等级 |
|------|------|---------|
| 硬图形不可导 | STE: 前向硬渲染，反向用模糊模板求梯度 | 🟢 低（diffbmp 已验证） |
| 多模板 GPU 存储 | 20 种 × 128² float = 1.3MB，CUDA texture array | 🟢 低 |
| 类型选择计算量 | 仅初始化+重定位时，~60K 次/周期，毫秒级 | 🟢 低 |
| FH6 Y 轴翻转 | 输出时统一处理 `-cy` | 🟢 低 |
| scale 坐标系差异 | 优化时用统一归一化坐标，输出时转换 | 🟡 中 |
| 3000 层硬图形质量 | 比纯椭圆大幅提升，但不如无限软椭圆 | 🟡 中 |

### 7.3 质量预期

对于 2000×2000 原图，3000 层，20 种图形，Python/PyTorch + Triton 实现：
- **预期 PSNR**: 25-32 dB（取决于原图复杂度）
- **视觉效果**: 作为游戏涂装远超手工，作为照片还原有可见损失
- **模糊影响**: σ=0.3 在画布上 < 0.5px，肉眼不可见
- **迭代速度**: Triton kernel ~手写 CUDA 95% 性能，每迭代 ~50-100ms（RTX 4090 估计）

---

## 8. 实现方案 (Python/PyTorch + Triton)

### 8.1 技术栈

| 层 | 技术 | 作用 |
|----|------|------|
| 渲染 kernel | **Triton** (改造 IGS `gs_triton_chunked.py`) | 前向 Over 合成 + 反向梯度 |
| 自动微分 | **PyTorch autograd** | 链式法则、参数更新（替代手写 CUDA backward） |
| 优化器 | **`torch.optim.Adam`** | 替代手写 CUDA Adam fused kernel |
| 损失函数 | **PyTorch** + diffbmp `LossRegistry` | MSE + Perceptual (VGG) |
| 模板采样 | **`F.grid_sample`** (或 Triton bilinear kernel) | 双线性采样模糊/硬模板 |
| 预处理 | OpenCV + scikit-learn K-means | 颜色量化、边缘检测 |
| FH6 导入 | **forza-painter-fh6** `fh6_typecode_import.py` | 直接 import 复用 |

### 8.2 项目结构

```
fh6_vectorizer/
├── pyproject.toml
├── src/
│   ├── renderer/
│   │   ├── triton_kernels.py     # Triton forward/backward Over compositing
│   │   ├── tile_renderer.py      # Tile-based renderer (参考 diffbmp SimpleTileRenderer)
│   │   └── templates.py          # 模板加载 + Gaussian blur
│   ├── optimizer/
│   │   ├── gradient_optimizer.py # Adam 优化循环 (torch.optim.Adam)
│   │   └── relocation.py         # 循环重定位逻辑 (移植 vinylizer)
│   ├── preprocess/
│   │   └── preprocessor.py       # 图像预处理
│   ├── output/
│   │   └── json_writer.py        # FH6 JSON 生成
│   └── templates/
│       └── generate.py           # 模板库生成器
├── scripts/
│   └── run_optimize.py           # CLI 入口
└── tests/
```

### 8.3 总体架构

```
┌──────────────────────────────────────────────────────────┐
│              离线预处理（一次性）                           │
│                                                          │
│  FH6 Vinyl Resources ──→ generate.py ──→ 模板库 (.pt)     │
│  (Vertices+Indices)     (OpenCV)          hard + soft    │
│                                                 │        │
│                                    硬模板 (前向)  软模板(反向) │
└──────────────────────────────────────────────────────────┘
                                                  │
┌─────────────────────────────────────────────────▼────────┐
│              Python/PyTorch + Triton 可微渲染管线            │
│                                                          │
│  1. Preprocessor: 加载原图, K-means, 边缘检测               │
│  2. 初始化: Importance Map 加权采样 → 每个位置试所有 type     │
│  3. 优化循环:                                             │
│     ┌─ Phase A: Adam 全局优化 (150步)                      │
│     │  前向: Triton kernel — 硬模板 Over 合成               │
│     │  反向: Triton kernel — 软模板 bilinear_sample 梯度    │
│     │  更新: torch.optim.Adam (替代手写 CUDA Adam)         │
│     ├─ Phase B: 扫描无用图形                               │
│     ├─ Phase C: 重定位 + 局部优化 (50步, frozen_mask)      │
│     └─ 重复 3 个大周期                                     │
│  4. 输出: JSON → forza-painter-fh6 直接注入                 │
└──────────────────────────────────────────────────────────┘
```

### 8.4 步骤拆分

#### Step 1: 模板库生成器 (Python, ~150 行)

**输入**: `E:\workspace\forza-painter-fh6\src\data\fh6_vinyl_resources\Vinyls\`

**输出**: 
- `templates_hard.pt` — `[20, 128, 128]` float32 tensor（硬模板，`{0,1}` 二值）
- `templates_soft.pt` — `[20, 128, 128]` float32 tensor（软模板，Gaussian blur σ=0.3）
- `type_map.json` — `{type_code → template_index}` 映射表

```python
import cv2, json, torch, numpy as np
from pathlib import Path

TEMPLATE_SIZE = 128
SIGMA = 0.3

def render_polygon(vertices, indices, size):
    """将 FH6 polygon 渲染为 bitmap"""
    canvas = np.zeros((size, size), dtype=np.float32)
    # vertices 需要从 FH6 坐标空间映射到 [0, size]
    pts = np.array([(v['X'], v['Y']) for v in vertices])
    # 归一化 + 缩放
    pts[:, 0] = (pts[:, 0] / 128.0 + 0.5) * size  # FH6 坐标系 [-128,128] → [0, size]
    pts[:, 1] = (pts[:, 1] / 128.0 + 0.5) * size
    triangles = np.array(indices).reshape(-1, 3)
    cv2.fillPoly(canvas, [pts[tri].astype(np.int32)], 1.0)
    return torch.from_numpy(canvas)

def gaussian_blur(tensor, sigma):
    """对 2D tensor 做高斯模糊"""
    arr = tensor.numpy()
    blurred = cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma)
    return torch.from_numpy(blurred)
```

#### Step 2: Triton 渲染 kernel (Python, ~200 行)

基于 IGS `gs_triton_chunked.py` 改造，核心改动点：

```python
# 改造前 (IGS 原始):
#   weight = exp(-0.5 * (cov00*dx² + cov11*dy² + 2*cov01*dx*dy)) * alpha
#   result = sum(weight * color) / (sum(weight) + eps)

# 改造后 (我们的版本):
#   alpha = bilinear_sample(templates[type], tex_x, tex_y) * opacity
#   result = Over_composite(result, alpha, color)  # 从后往前累乘 T

@triton.jit
def over_composite_forward_kernel_chunked(
    # ... 参数类似 IGS，但使用 templates 3D tensor 替代 cov_inv ...
    templates_hard_ptr,  # [num_types, T, T] 硬模板
    templates_soft_ptr,  # [num_types, T, T] 软模板
    type_indices_ptr,    # [N] 每个 primitive 的 type
    # ...
):
    # 1. 加载 chunk 内所有 primitive 参数
    # 2. 对每个像素 tile:
    #    - 坐标变换到模板空间
    #    - 硬模板 bilinear_sample → alpha（前向）
    #    - Over 合成: C += T * alpha * color; T *= (1-alpha)
    # 3. 保存 T 和 alpha 到 scratch 用于反向
    pass

@triton.jit  
def over_composite_backward_kernel_chunked(
    templates_soft_ptr,  # 反向只用软模板
    # ...
):
    # 用软模板重新计算 forward → 求梯度
    # 梯度通过 bilinear_sample 的 autograd 自动回传
    pass
```

**关键设计**：
- 前向用硬模板（`alpha > 0.5 ? 1 : 0`），匹配 FH6 游戏内效果
- 反向用软模板（连续值），确保梯度可流动（STE 策略）
- Chunked 结构 + shared memory 归约全部保留自 IGS

#### Step 3: 优化器 (Python, ~300 行)

```python
import torch
from torch.optim import Adam

class GradientOptimizer:
    def __init__(self, config):
        # 9 个连续优化参数: cx, cy, rx, ry, angle, R, G, B, opacity
        # type_code 是离散参数，不参与梯度优化
        self.params = torch.nn.ParameterDict({
            'cx':      nn.Parameter(torch.zeros(N)),
            'cy':      nn.Parameter(torch.zeros(N)),
            'rx':      nn.Parameter(torch.rand(N) * 10 + 2),
            'ry':      nn.Parameter(torch.rand(N) * 10 + 2),
            'angle':   nn.Parameter(torch.rand(N) * 360),
            'color_r': nn.Parameter(torch.rand(N) * 255),
            'color_g': nn.Parameter(torch.rand(N) * 255),
            'color_b': nn.Parameter(torch.rand(N) * 255),
            'opacity': nn.Parameter(torch.ones(N)),
        })
        self.type_codes = torch.zeros(N, dtype=torch.long)  # 离散参数
        
        # PyTorch Adam 替代手写 CUDA Adam
        self.optimizer = Adam(self.params.values(), lr=0.1)
        
        # Cosine annealing scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_iters)
    
    def optimize_step(self, target, mask):
        self.optimizer.zero_grad()
        
        # 前向: Triton kernel → 渲染图
        rendered = render_over_composite(
            self.params, self.type_codes, self.templates_hard)
        
        # 损失: MSE + Perceptual
        mse = F.mse_loss(rendered, target)
        perceptual = perceptual_loss(rendered, target, self.vgg)
        loss = mse + 0.2 * perceptual
        
        # 反向: PyTorch autograd 自动处理
        loss.backward()
        
        # STE: 颜色量化到整数（前向用量化值，反向梯度不变）
        with torch.no_grad():
            self.params['color_r'].data = self.params['color_r'].round().clamp(0, 255)
        
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()
```

#### Step 4: JSON 输出 + FH6 注入 (Python, ~50 行)

```python
# 直接复用 forza-painter-fh6 的导入逻辑
import json

def generate_fh6_json(shapes, output_path):
    j = {"shapes": []}
    for s in shapes:
        shape_word = (s.type_code - 0x100000) & 0xFFFF
        j["shapes"].append({
            "type": s.type_code,
            "data": [s.cx, -s.cy,                # FH6 Y 翻转
                     s.rx / DIVISOR, s.ry / DIVISOR,
                     (360 - s.angle) % 360,       # FH6 角度方向
                     0.0],                         # skew
            "color": [s.r, s.g, s.b, s.a],
            "mask": 0
        })
    with open(output_path, 'w') as f:
        json.dump(j, f)

# 注入游戏:
# python fh6_typecode_import.py --json output.json --layers 3000 --pid <PID>
```

### 8.5 关键技术决策

1. **模板大小**: 128×128（平衡精度与显存）
2. **模糊 σ**: 0.3（在 128px 模板上 ≈ 1px 过渡，画布上完全不可见）
3. **STE 策略**: 前向硬模板 + 反向软模板（参考 diffbmp 的 `S_blurred` vs `S`）
4. **类型选择时机**: 仅在初始化和重定位时（优化过程中固定）
5. **模板存储**: `torch.Tensor` `[num_types, T, T]` + `F.grid_sample` 或 Triton bilinear kernel
6. **颜色空间**: sRGB→Linear→sRGB（与 vinylizer/diffbmp 一致）
7. **优化器**: `torch.optim.Adam` + `CosineAnnealingLR`（替代手写 CUDA Adam，代码减少 ~200 行）

### 8.6 代码复用清单（Python 版）

| 来源 | 复用内容 | 方式 |
|------|---------|------|
| **diffbmp** | `VectorRenderer`, `SimpleTileRenderer` 框架 | 直接 import 或 fork |
| **diffbmp** | `LossRegistry` (MSE + Perceptual) | 直接 import |
| **IGS** | `gs_triton_chunked.py` kernel 结构 | 改造（Gaussian → bilinear_sample + Over） |
| **forza-painter** | `fh6_typecode_import.py` | 直接 import |
| **vinylizer** | 循环重定位算法 | 移植到 Python（~200 行） |
| **vinylizer** | STE 策略 | 逻辑移植（~50 行） |

### 8.7 Python 版 vs C++ 版对比

| 维度 | C++/CUDA 版 | Python/PyTorch+Triton 版 |
|------|------------|------------------------|
| 开发效率 | 低（手写 CUDA、手推导数） | **高（autograd + Triton）** |
| AI 辅助成本 | 高（C++ token 多） | **低（Python 简洁）** |
| Perceptual Loss | 需 OpenCV DNN (~200行) | **开箱即用 (~5行)** |
| Adam 优化器 | 手写 CUDA fused kernel (~150行) | **`torch.optim.Adam` (~2行)** |
| Kernel 性能 | 最优（手写 CUDA） | **接近（Triton 自动调优）** |
| 调试难度 | 高（cuda-gdb） | **低（Python debugger）** |
| 代码总量 | ~2000 行 C++/CUDA | **~800 行 Python** |

---

## 9. 图形精选策略

### 9.1 推荐列表 (15-20 种)

基于"不可替代性"原则精选：

```
必备基础几何 (6):
  type_code  名称      用途
  ─────────  ────────  ──────────────────
  1048677    Square    大面积纯色填充
  1048678    Circle    圆形区域
  1048679    Triangle  尖锐特征
  1048712    Ellipse   rx≠ry 场景
  1048777+   Gradient  原生软渐变 (FH6独有!)
  1048688    Circle    镂空/环形
             Border

高收益 (8):
  菱形、五边形/六边形、星形、新月(Crescent)
  条纹 1-2 种、十字形、水滴

按需 (6):
  心形、火焰、大写字母 1 组 (仅文字涂装)
```

### 9.2 边际收益分析

- 1 种 → 8 种: 表达能力提升约 3-5×
- 8 种 → 20 种: 提升约 1.5-2×
- 20 种 → 1000 种: 提升约 5-15%（边际递减）

计算开销：20 种在初始化时只需 ~60K 次比较（毫秒级），1000 种需要 ~3M 次（百毫秒级但可接受）。

---

## 10. 潜在风险与缓解

| 风险 | 概率 | 缓解方案 |
|------|------|---------|
| FH6 scale 坐标系映射错误 | 中 | 参考 `fh6_typecode_export.py` 的 decode 逻辑，直接对比已知数据验证 |
| STE 前向/反向不一致导致收敛差 | 低 | diffbmp 已验证此方法；sigma 取极小值确保差距 < 0.5px |
| 硬图形叠加出现可见接缝 | 中 | 引入少量 Gradient Shapes 作为"胶水层"填充接缝 |
| 3000 层 GPU 显存不足 | 低 | 128² 模板仅 1.3MB；梯度缓冲区与 diffbmp 一致 |
| 模板渲染器未正确处理 VerticesAlpha | 中 | 大部分图形的 VerticesAlpha 是全 255（不透明），少数需特殊处理 |
| **Triton kernel Over 合成改造出错** | 中 | IGS 原版是归一化加权和，改为 Over 需要重新验证 backward 梯度正确性（先写 PyTorch 纯 Python 版做 golden reference） |
| **Python overhead 导致迭代变慢** | 低 | 核心计算在 Triton/CUDA kernel 内；Python 只做调度，overhead 可忽略 |
| **Triton 兼容性（特定 GPU 不支持）** | 低 | Triton 支持 CUDA 8.0+，覆盖 GTX 10 系列及以上所有 NVIDIA GPU |

---

## 附录 A: 进一步参考资料

### 论文
- DiffBMP: https://arxiv.org/abs/2602.22625 — Section 3.2 "Soft Rasterization via Gaussian Blur"
- DiffVG (diffbmp 的前身参考): https://github.com/BachiLi/diffvg

### 代码参考
- vinylizer 反向传播链式法则: `e:\workspace\vinylizer\src\cuda\render_kernel.cu` 第 190-300 行（参考数学推导）
- diffbmp Python 框架: `E:\workspace\diffbmp\pydiffbmp\core\renderer\vector_renderer.py` — `VectorRenderer` 完整类
- diffbmp Loss: `E:\workspace\diffbmp\pydiffbmp\util\loss_functions.py` — `LossRegistry`（可直接 import）
- IGS Triton kernel: `E:\workspace\IGS\src\igs\gs_triton_chunked.py` — Chunked forward kernel（改造模板）
- FH6 内存结构: `E:\workspace\forza-painter-fh6\src\fh6_typecode_import.py` — `decode()` + 导入逻辑（可直接 import）
- FH6 图形家族列表: `E:\workspace\forza-painter-fh6\src\fh6_vinyl_resources.py` `VINYL_TYPE_BASES`

### Python 生态参考
- `torchvision.models.vgg16(weights='IMAGENET1K_V1')` — Perceptual Loss 预训练模型
- `torch.optim.Adam` + `torch.optim.lr_scheduler.CosineAnnealingLR` — 替代手写 CUDA Adam
- `F.grid_sample` — 双线性采样（反向自动可微）
- `triton.autotune` — 自动调优 block size
- `cv2.GaussianBlur` — 模板预处理

### 模板生成参考
- FH6 图形几何格式: `E:\workspace\forza-painter-fh6\src\data\fh6_vinyl_resources\Vinyls\Primitives\1`
- 字体注册表: `E:\workspace\forza-painter-fh6\src\data\fh6_font_registry.json`
- 图形目录 CSV: `E:\workspace\forza-painter-fh6\src\data\FH6 Shape Library Data - FH6 Shape Library Data.csv`
