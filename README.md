# FH6 多图形可微矢量化工具

使用 **可微渲染** 将任意图片重建成 FH6（Forza Horizon 6）涂装图形组合。

融合 [diffbmp](https://github.com/smhongok/diffbmp) (CVPR 2026)、
[vinylizer](https://github.com/Heavenchaos/vinylizer)、
[IGS](https://github.com/KohakuBlueleaf/IGS) 的核心思想。

## 安装

```bash
pip install -e .

# GPU 加速（Windows + RTX 4090 验证）
pip install triton-windows
```

## 使用

5 种数学渐变图形（椭圆 + 4 种方向渐变矩形），连续 alpha 渲染，无需 FH6 数据依赖：

```bash
python scripts/run_optimize.py \
    -i photo.png \
    -o result.png \
    -n 1000 \
    --size 512 512 \
    --device cuda \
    --cycles 3 \
    --global-steps 100 \
    --local-steps 30 \
    --l1-weight 0.1 \
    --lr 0.05 \
    --template-cache templates5.pt
```

首次运行会生成模板缓存 `templates5.pt`，后续自动复用。

## 项目结构

```
multi-shape-vectorizer/
├── src/fh6_vectorizer/
│   ├── templates.py          # 5 种数学渐变模板
│   ├── ste_renderer.py       # 连续 alpha Over 合成渲染器（PyTorch + Triton）
│   ├── optimizer.py          # Adam 优化 + 循环重定位
│   ├── loss.py               # MSE / L1 / Huber / 感知 / 灰度 / Alpha 正则
│   ├── preprocess.py         # 重要性图
│   ├── json_writer.py        # FH6 JSON 导出
│   └── pipeline.py           # 端到端流水线
├── scripts/
│   ├── run_optimize.py       # CLI 入口
│   ├── visualize_templates.py
│   └── plot_loss.py
└── tests/
```

## 参考

| 项目 | 说明 |
|------|------|
| [diffbmp](https://github.com/smhongok/diffbmp) | CVPR 2026 — 多图形可微渲染框架 |
| [vinylizer](https://github.com/Heavenchaos/vinylizer) | Over 合成 + 循环重定位 |
| [IGS](https://github.com/KohakuBlueleaf/IGS) | Triton kernel 模式 |

## 许可

MIT

