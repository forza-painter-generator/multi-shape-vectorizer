# FH6 多图形可微矢量化工具

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-red)](https://pytorch.org/)
[![Triton](https://img.shields.io/badge/triton-3.7-green)](https://github.com/triton-lang/triton)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

使用 **可微渲染** 技术，自动将任意图片重建成 FH6（Forza Horizon 6 / 极限竞速：地平线 6）
涂装图形组合。融合了 [diffbmp](https://github.com/smhongok/diffbmp) (CVPR 2026)、
[vinylizer](https://github.com/Heavenchaos/vinylizer)、
[IGS](https://github.com/KohakuBlueleaf/IGS) 和
[forza-painter-fh6](https://github.com/bvzrays/forza-painter-fh6) 的核心思想。

## 架构

```
输入图像 → 预处理 → 重要性图
                        ↓
模板库 (硬+软) ──→ STE Over 合成渲染器 ←── Adam 优化器
                        ↓                        ↑
                  渲染结果 ──→ MSE + 感知损失
                        ↓
                  循环重定位
                        ↓
                  FH6 JSON 输出
```

## 核心特性

- **STE（直通估计器）**：前向渲染使用硬/二值图形（匹配游戏内效果），反向传播使用软/模糊图形传递梯度。
- **多图形支持**：16 种合成几何模板（圆、方、三角、星、爱心、月牙等），或真实 FH6 多边形数据。
- **线性空间 Over 合成**：物理正确的 Porter-Duff alpha 混合 + sRGB ↔ Linear 转换。
- **循环重定位**：自动识别"无用"图形，将其移动到误差最大的区域（来自 vinylizer）。
- **丰富的损失函数**：MSE、L1、Huber、灰度 MSE、VGG 感知损失、透明度正则化。
- **重要性图初始化**：Canny 边缘 + 颜色方差引导初始图形分布。
- **FH6 JSON 导出**：生成的 JSON 可直接导入 Forza Horizon 6。
- **分块渲染**：大画布（≥256px）自动启用 tile-based 渲染。
- **GPU 加速**：支持 CUDA（RTX 4090 测试通过）+ Triton（通过 `triton-windows` + `torch.compile`）。

## 安装

```bash
# 克隆仓库
git clone <repo-url>
cd multi-shape-vectorizer

# 安装
pip install -e .

# GPU 加速（Windows + RTX 4090 验证）
pip install triton-windows

# 可选：感知损失
pip install torchvision

# 可选：loss 曲线绘图
pip install matplotlib
```

## 快速开始

```bash
# 基础：200 个图形，256×256，3 轮优化，MSE 损失
python scripts/run_optimize.py -i photo.jpg -o result.png -n 200 --size 256 256

# 高质量：500 个图形，感知损失，重要性采样，FH6 JSON
python scripts/run_optimize.py -i photo.jpg -o result.png \
    -n 500 --size 512 512 --cycles 5 --perceptual \
    --l1-weight 0.1 --alpha-reg 0.01 --fh6-json

# 使用 JSON 配置文件
python scripts/run_optimize.py -i photo.jpg -o result.png \
    --config-file configs/default.json

# 使用真实 FH6 图形数据
python scripts/run_optimize.py -i photo.jpg -o result.png \
    --fh6-data "path/to/Vinyls/" --template-cache templates.pt

# GPU 加速 + 中间快照
python scripts/run_optimize.py -i photo.jpg -o result.png \
    -n 500 --size 512 512 --device cuda --snapshot-dir snapshots/
```

## 工具脚本

```bash
# 可视化模板库
python scripts/visualize_templates.py -n 16 -o templates.png

# 绘制 loss 曲线
python scripts/plot_loss.py result.history.json -o loss_plot.png

# GPU 性能基准测试
python scripts/benchmark.py
python scripts/benchmark_triton.py

# torch.profiler 性能分析
python scripts/profiler.py --size 256 --num-shapes 100 --device cuda
```

## 项目结构

```
multi-shape-vectorizer/
├── configs/
│   └── default.json              # 默认超参数配置
├── src/fh6_vectorizer/
│   ├── templates.py              # 模板生成（合成 + FH6）
│   ├── ste_renderer.py           # STE Over 合成渲染器（+ 分块）
│   ├── optimizer.py              # Adam + 循环重定位 + 回滚
│   ├── loss.py                   # MSE / L1 / Huber / 感知 / 灰度 / Alpha 正则
│   ├── preprocess.py             # K-means / Canny / 重要性图
│   ├── json_writer.py            # FH6 JSON 导出
│   ├── triton_kernels.py         # Triton 加速 kernel
│   ├── cuda_renderer.py          # CUDA C++ extension kernel
│   └── pipeline.py               # 端到端流水线
├── scripts/
│   ├── run_optimize.py           # CLI 入口
│   ├── plot_loss.py              # Loss 曲线可视化
│   ├── visualize_templates.py    # 模板 PNG 网格
│   ├── benchmark.py              # CPU/GPU 性能基准
│   ├── benchmark_triton.py       # Triton 加速对比
│   └── profiler.py               # torch.profiler 分析
├── tests/
│   ├── test_core.py              # 核心单元测试
│   ├── test_new_features.py      # 新特性集成测试
│   └── test_extra.py             # 扩展测试（17 项）
├── TASKS.md                      # 详细任务清单
├── IMPLEMENTATION_PLAN.md        # 完整技术方案
├── CHANGELOG.md                  # 变更记录
├── LICENSE                       # MIT 许可
└── pyproject.toml
```

## 参考项目

| 项目 | 说明 |
|------|------|
| [diffbmp](https://github.com/smhongok/diffbmp) | CVPR 2026 — Gaussian Blur 软光栅化、多图形可微渲染框架 |
| [vinylizer](https://github.com/Heavenchaos/vinylizer) | Over 合成、循环重定位、STE 策略、CUDA 融合 kernel |
| [IGS](https://github.com/KohakuBlueleaf/IGS) | 2D Gaussian Splatting、Triton chunked kernel 模式 |
| [forza-painter-fh6](https://github.com/bvzrays/forza-painter-fh6) | FH6 图形几何数据、导入/导出管线 |

## 致谢

本项目从方案设计（[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)）到代码实现，
全程由 **DeepSeek** 辅助完成。从最初的技术调研（对比 vinylizer / diffbmp / IGS /
forza-painter-fh6 四个项目）、架构设计（STE + Over 合成 + 循环重定位）、
到逐行代码编写、测试修复、GPU 适配，DeepSeek 提供了持续且关键的支持。

恩情还不完 🙏

## 许可

MIT — 详见 [LICENSE](LICENSE)
