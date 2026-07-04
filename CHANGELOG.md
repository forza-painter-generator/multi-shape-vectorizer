# 变更记录

## [0.2.0] — 2026-07-04

### 新增

- **16 种合成模板**: 五边形、六边形、月牙、爱心、箭头、水滴、V 形、四角星
- **sRGB ↔ Linear 颜色空间转换** — Over 合成中使用线性空间，物理正确
- **L1 / Huber / 灰度 MSE / Alpha 正则化** 损失函数
- **重要性图初始化** — Canny 边缘 + 颜色方差引导初始图形分布
- **目标图像取色** — 从原图双线性采样初始颜色 + 小噪声
- **智能类型选择** — 重定位时尝试所有模板类型，选择局部 MSE 最低的
- **FH6 JSON 输出** (`json_writer.py`) — Y 轴翻转、角度反转、uint8 颜色
- **图像预处理模块** (`preprocess.py`) — K-means 量化、Canny 边缘、重要性图
- **分块渲染** — AABB 裁剪，画布 ≥256px 自动启用
- **重定位 MSE 回滚** — Phase C 后 MSE 恶化则恢复参数快照
- **Loss 历史日志** — 自动保存 `.history.json`
- **中间快照** — 优化过程中每 N 步保存渲染 PNG
- **`scripts/plot_loss.py`** — loss 曲线可视化
- **`scripts/visualize_templates.py`** — 模板 PNG 网格导出
- **`scripts/profiler.py`** — torch.profiler 性能分析
- **`scripts/benchmark.py`** — 多尺寸/多形状性能基准
- **`configs/default.json`** — JSON 配置文件，6 个配置组
- **26 项单元测试** — `test_core.py` / `test_new_features.py` / `test_extra.py`
- **`README.md`** — 完整中文项目文档

### 变更

- Over 合成改为线性颜色空间
- `GradientOptimizer` 支持所有新损失函数权重和配置项
- `STEVectorRenderer.forward()` 自动选择分块/非分块渲染

### GPU + Triton

- **Triton 支持** — 通过 `triton-windows` 3.7.1 + `torch.compile`
- **RTX 4090 验证** — CUDA 12.9，24GB VRAM
- **GPU 加速 1.1-1.2×** — torch.compile 自动生成 Triton kernel
- **`triton_kernels.py`** — 手写 Triton Over 合成 kernel（forward-only）
- **`cuda_renderer.py`** — CUDA C++ extension 备选方案

## [0.1.0] — 2026-07-03

### 新增

- 核心 STE 可微渲染管线
- Porter-Duff Over 合成 + 循环重定位
- 8 种合成几何模板（圆、方、三角、椭圆、菱形、星、十字、环）
- MSE + VGG 感知损失函数
- Adam 优化器 + 余弦退火
- FH6 模板库构建器（从 forza-painter-fh6 数据）
- CLI 入口 (`scripts/run_optimize.py`)
- 基础单元测试
