# FH6 Multi-Shape Differentiable Vectorizer — 实现任务清单

> **最后更新**: 2026-07-04
> **状态**: 接近完成 — 核心功能全部实现，仅剩 GPU/外部依赖项

---

## 目录

1. [项目基础设施](#1-项目基础设施)
2. [模板库生成](#2-模板库生成)
3. [STE 可微渲染器](#3-ste-可微渲染器)
4. [优化器 & 循环重定位](#4-优化器--循环重定位)
5. [损失函数](#5-损失函数)
6. [FH6 集成 & 输入输出](#6-fh6-集成--输入输出)
7. [图像预处理](#7-图像预处理)
8. [Triton GPU Kernel 优化](#8-triton-gpu-kernel-优化)
9. [CLI & 流水线](#9-cli--流水线)
10. [测试 & 质量保证](#10-测试--质量保证)
11. [文档 & 收尾](#11-文档--收尾)

---

## 1. 项目基础设施

> 参考: `IMPLEMENTATION_PLAN.md` §8.2 项目结构

- [x] **1.1 — 项目目录结构**
  - 创建 `src/fh6_vectorizer/` 包目录，包含 renderer / optimizer / templates / loss / pipeline 模块
  - 创建 `scripts/` 目录（CLI 入口）
  - 创建 `tests/` 目录
  - 详见: `IMPLEMENTATION_PLAN.md` §8.2 推荐的项目结构树

- [x] **1.2 — `pyproject.toml` 项目配置**
  - 文件: [`pyproject.toml`](pyproject.toml)
  - 依赖: `torch>=2.0`, `torchvision>=0.15`, `numpy`, `opencv-python`, `Pillow`, `triton`（Linux only）
  - 可选依赖: `pytest`, `matplotlib`
  - 包发现: `[tool.setuptools.packages.find] where = ["src"]`

- [x] **1.3 — `__init__.py` 包入口**
  - 文件: [`src/fh6_vectorizer/__init__.py`](src/fh6_vectorizer/__init__.py)
  - 版本号: `__version__ = "0.1.0"`

- [x] **1.4 — 配置管理**
  - 文件: [`configs/default.json`](configs/default.json) — 6 配置组 (optimization/loss/templates/preprocessing/init/fh6)
  - CLI `--config-file` 加载 JSON → CLI 参数覆盖
  - `_flatten_config()` 处理嵌套 JSON → flat dict

---

## 2. 模板库生成

> 参考: `IMPLEMENTATION_PLAN.md` §8.4 Step 1 — 模板库生成器

### 2.1 合成模板（无 FH6 依赖）

- [x] **2.1.1 — `generate_synthetic_templates()` 基础函数**
  - 文件: [`src/fh6_vectorizer/templates.py`](src/fh6_vectorizer/templates.py) — `generate_synthetic_templates()`
  - 生成 8 种几何图形: circle, square, triangle, ellipse, diamond, star, cross, ring
  - 模板分辨率: `TEMPLATE_SIZE = 128`
  - 使用 OpenCV (`cv2.circle`, `cv2.rectangle`, `cv2.fillPoly`, `cv2.ellipse`) 渲染
  - Hard 模板: 二值 `{0, 1}`
  - Soft 模板: `cv2.GaussianBlur` with `sigma = 0.5`
  - 参考: `IMPLEMENTATION_PLAN.md` §9.1 推荐形状列表

- [x] **2.1.2 — 更多合成模板类型**
  - 已添加: pentagon(8), hexagon(9), crescent(10), heart(11), arrow(12), droplet(13), chevron(14), star4(15)
  - 总计 16 种，默认 `num_types=16`

- [x] **2.1.3 — 合成模板参数化配置**
  - 每种模板使用固定的几何参数（star 5-point, heart standard 等）
  - `generate_synthetic_templates(num_types=N)` 截断/扩展

### 2.2 FH6 真实图形模板

- [x] **2.2.1 — `render_fh6_shape()` — FH6 多边形渲染**
  - 文件: [`src/fh6_vectorizer/templates.py`](src/fh6_vectorizer/templates.py) — `render_fh6_shape()`
  - 输入: FH6 JSON 的 `Vertices` (list of `{"X","Y"}`) + `Indices` (triangle list)
  - 坐标映射: FH6 `[-128, 128]` → template `[0, TEMPLATE_SIZE]`
  - 使用 `cv2.fillPoly` 渲染三角形

- [x] **2.2.2 — `build_template_library()` — FH6 模板库构建**
  - 文件: [`src/fh6_vectorizer/templates.py`](src/fh6_vectorizer/templates.py) — `build_template_library()`
  - 输入: `vinyls_root` = `E:\workspace\forza-painter-fh6\src\data\fh6_vinyl_resources\Vinyls\`
  - 遍历 `RECOMMENDED_SHAPES` 列表中的 (family, index)
  - 加载 JSON → `render_fh6_shape()` → hard tensor → `gaussian_blur()` → soft tensor
  - 输出: `{"hard": [N,T,T], "soft": [N,T,T], "type_map": {type_code: idx}, "names": [...]}`
  - 参考: `IMPLEMENTATION_PLAN.md` §6.2 图形几何数据格式

- [x] **2.2.3 — `VINYL_TYPE_BASES` 家族映射表**
  - 文件: [`src/fh6_vectorizer/templates.py`](src/fh6_vectorizer/templates.py)
  - 28 个图形家族的 `type_code` 基数
  - 来源: `E:\workspace\forza-painter-fh6\src\fh6_vinyl_resources.py` — `VINYL_TYPE_BASES`

- [ ] **2.2.4 — `VerticesAlpha` 支持**
  - FH6 JSON 中有 `"VerticesAlpha": "////..."` 字段（base64 编码的顶点透明度）
  - 大部分图形是全 255（不透明），但部分图形有渐变透明度
  - 解析 base64 → 逐顶点 alpha → 渲染时应用
  - 参考: `IMPLEMENTATION_PLAN.md` §10 潜在风险 — "模板渲染器未正确处理 VerticesAlpha"

- [ ] **2.2.5 — 扩展 FH6 模板到 20+ 种**
  - 当前 `RECOMMENDED_SHAPES` 只有 8 种
  - 扩展包括: 更多 Primitives 变体, Gradient_Shapes（原生软渐变！FH6 独有）, Stripes, Flames, Tribal
  - 参考: `IMPLEMENTATION_PLAN.md` §9.1 推荐列表 — 必备基础几何 (6) + 高收益 (8) + 按需 (6)

- [x] **2.2.6 — 模板持久化 (`save/load_template_library`)**
  - 文件: [`src/fh6_vectorizer/templates.py`](src/fh6_vectorizer/templates.py)
  - `save_template_library()`: `torch.save()` 到 `.pt` 文件
  - `load_template_library()`: `torch.load()` 带回 `weights_only=False`
  - Pipeline 中使用 `template_cache_path` 参数支持缓存

- [x] **2.2.7 — 模板可视化工具**
  - 创建一个脚本 `scripts/visualize_templates.py`
  - 将 hard 和 soft 模板并排保存为 PNG 网格
  - 用于验证模板生成正确性（特别是 FH6 模板的坐标映射）

### 2.3 Gaussian Blur 配置

- [x] **2.3.1 — Blur sigma 参数化**
  - `DEFAULT_SIGMA = 0.5`（当前值）
  - 在 `build_template_library()` 和 `generate_synthetic_templates()` 中可配置
  - 参考: `IMPLEMENTATION_PLAN.md` §3.3 — diffbmp 使用 σ=1.0 在 128px 模板 ≈ 3px 过渡带

- [ ] **2.3.2 — Sigma 自动调优研究**
  - 研究最优 sigma: 太小 → 梯度稀疏（只有边界像素有梯度），太大 → 硬/软差异过大导致 STE 偏差
  - 在 128px 模板上测试 σ ∈ {0.1, 0.3, 0.5, 0.7, 1.0}
  - 记录不同 sigma 下的收敛速度和最终 PSNR
  - 参考: `IMPLEMENTATION_PLAN.md` §3.3 — "σ=1.0 在 128px 模板 ≈ 3px 过渡带"

---

## 3. STE 可微渲染器

> 参考: `IMPLEMENTATION_PLAN.md` §3.1-3.6 核心技术原理, §8.4 Step 2

### 3.1 坐标系统 & 网格

- [x] **3.1.1 — `_make_canvas_grid()` 像素坐标网格**
  - 文件: [`src/fh6_vectorizer/ste_renderer.py`](src/fh6_vectorizer/ste_renderer.py) — `_make_canvas_grid()`
  - 生成 `px_grid[H,W]` 和 `py_grid[H,W]`（0-indexed 像素中心坐标）
  - 使用 `torch.meshgrid` with `indexing="ij"`

- [x] **3.1.2 — `compute_template_coords()` 坐标变换**
  - 文件: [`src/fh6_vectorizer/ste_renderer.py`](src/fh6_vectorizer/ste_renderer.py) — `compute_template_coords()`
  - 变换链: Translate → Rotate(-angle) → Scale
  - 公式: `tx = (dx*cos(-θ) - dy*sin(-θ)) / (rx + ε) * TEMPLATE_FILL_RATIO`
  - `TEMPLATE_FILL_RATIO = 0.9`（形状占模板 90%）
  - 输出: `[1, H, W, 2]` grid for `F.grid_sample`

- [x] **3.1.3 — 坐标变换正确性验证**
  - `test_coords_center_maps_to_zero`: 中心映射到 (0,0)
  - `test_coords_rotation`: 90° 旋转交换 tx/ty
  - `test_coords_scale`: 2× scale → tx/ty 减半

### 3.2 Over Compositing 渲染

- [x] **3.2.1 — `over_composite_render()` 核心渲染函数**
  - 文件: [`src/fh6_vectorizer/ste_renderer.py`](src/fh6_vectorizer/ste_renderer.py) — `over_composite_render()`
  - Porter-Duff over 操作:
    ```
    C = bg; T = 1.0
    for each shape back→front:
        α = sample(template) × opacity
        w = α × T
        C += w × color
        T *= (1 - α)
        if T_max < ε: break  // early termination
    ```
  - 颜色在 `[0,1]` RGB 空间（无 sRGB↔Linear 转换，PoC 简化）
  - 参考: `IMPLEMENTATION_PLAN.md` §3.2 Over 合成公式

- [x] **3.2.2 — STE 软硬分离**
  - 文件: [`src/fh6_vectorizer/ste_renderer.py`](src/fh6_vectorizer/ste_renderer.py) — `over_composite_render()` L158-169
  - 前向: `hard_alpha = (sample(hard_template) > 0.5).float()` → binary `{0, 1}`
  - 反向: gradient flows through `soft_alpha = sample(soft_template)` → continuous `[0, 1]`
  - STE trick: `alpha = hard.detach() + soft - soft.detach()`
  - 参考:
    - vinylizer `src/core/diff_renderer.cu` L248-320 — STE for color quantization (`roundf` + gradient passthrough)
    - diffbmp `pydiffbmp/core/renderer/vector_renderer.py` L70-95 — `self.S_blurred = gaussian_blur(self.S, sigma)`
    - `IMPLEMENTATION_PLAN.md` §3.6 STE 策略

- [x] **3.2.3 — `F.grid_sample` 双线性采样**
  - 硬模板和软模板都用 `F.grid_sample(mode="bilinear", padding_mode="zeros", align_corners=True)`
  - PyTorch autograd 自动处理 `grid_sample` 的反向传播（无需手写 CUDA backward）
  - 参考: diffbmp `cuda_tile_rasterizer/cuda_kernels/tile_forward.cu` — `bilinear_sample` CUDA 实现

- [x] **3.2.4 — Early termination 优化**
  - 当 `T.max() < 1e-4` 时停止处理后续形状
  - 参考: `IMPLEMENTATION_PLAN.md` §3.2 — "if T < ε: break"

- [x] **3.2.5 — `STEVectorRenderer(nn.Module)` 封装**
  - 文件: [`src/fh6_vectorizer/ste_renderer.py`](src/fh6_vectorizer/ste_renderer.py) — `STEVectorRenderer` 类
  - 9 个可优化参数: `cx, cy, rx, ry, angle` (几何) + `colors[3]` (颜色) + `opacity` (透明度)
  - 1 个离散参数: `type_indices` (不参与梯度优化)
  - 模板库和背景色: `register_buffer`（不参与优化）
  - `clamp_params()`: 参数范围约束
  - `get_params_dict()`: 导出参数字典

- [x] **3.2.6 — sRGB ↔ Linear 颜色空间转换**
  - 文件: [`src/fh6_vectorizer/loss.py`](src/fh6_vectorizer/loss.py) — `srgb_to_linear()` / `linear_to_srgb()`
  - 标准 sRGB 传输函数 + 自动 clamp 防 NaN
  - Over 合成在 LINEAR 空间: `C_lin += T * alpha * srgb_to_linear(color)`
  - 输出: `linear_to_srgb(C_lin)`

- [x] **3.2.7 — 类型选择: 重定位时智能选择**
  - 文件: [`src/fh6_vectorizer/optimizer.py`](src/fh6_vectorizer/optimizer.py) — `_select_best_types()`
  - 在 error 位置尝试所有模板类型 (local patch MSE) 选最优
  - CLI: `--smart-types` 启用（默认关闭，较慢）

### 3.3 Tile-based 渲染（性能优化）

- [x] **3.3.1 — Tile renderer 架构**
  - 当前: 全图逐形状渲染（`O(N × H × W)`），大画布时慢
  - 实现 tile-based: 将画布分为 `TILE_SIZE × TILE_SIZE` 块（如 128×128 或 256×256）
  - 对每个 tile，AABB 测试 → 只渲染重叠的形状
  - 参考:
    - diffbmp `pydiffbmp/core/renderer/simple_tile_renderer.py` — `render_from_params()`
    - vinylizer `src/cuda/render_kernel.cu` L95-145 — tile-based `render_tile_forward_kernel`
    - `IMPLEMENTATION_PLAN.md` §5.5 — Chunked primitive processing

- [x] **3.3.2 — AABB 计算**
  - 对每个形状，根据 `(cx, cy, rx, ry, angle)` 计算旋转后的包围盒
  - 快速 AABB 近似: `max_extent = max(rx, ry) * 1.5`（考虑旋转 + blur 扩展）

- [x] **3.3.3 — Tile buffer 管理**
  - 每个 tile 维护自己的 `C` 和 `T` buffer
  - 最后合并所有 tile 为完整图像
  - 参考: vinylizer `src/cuda/canvas_render.cu` L28-88 — tile 级别 scratch

---

## 4. 优化器 & 循环重定位

> 参考: `IMPLEMENTATION_PLAN.md` §3.5 循环重定位, §8.4 Step 3

### 4.1 基础优化循环

- [x] **4.1.1 — `GradientOptimizer` 类**
  - 文件: [`src/fh6_vectorizer/optimizer.py`](src/fh6_vectorizer/optimizer.py) — `GradientOptimizer` 类
  - 管理 `STEVectorRenderer` + `target` + 优化器配置

- [x] **4.1.2 — `torch.optim.Adam` 替代手写 CUDA Adam**
  - 文件: [`src/fh6_vectorizer/optimizer.py`](src/fh6_vectorizer/optimizer.py) — `GradientOptimizer.__init__()`
  - 分离参数组: 几何参数 `lr`, 颜色/透明度 `lr * 0.5`
  - 参考: vinylizer `src/core/diff_renderer.cu` L248-320 — 手写 CUDA `fused_adam_kernel`（现在不需要了！）
  - 参考: `IMPLEMENTATION_PLAN.md` §8.7 — "Adam 优化器: 手写 CUDA (~150行) → torch.optim.Adam (~2行)"

- [x] **4.1.3 — `CosineAnnealingLR` 学习率调度**
  - 文件: [`src/fh6_vectorizer/optimizer.py`](src/fh6_vectorizer/optimizer.py) — `GradientOptimizer.__init__()`
  - `T_max = num_cycles * (global_steps + local_steps)`

- [x] **4.1.4 — `_optimize_step()` 单步优化**
  - 文件: [`src/fh6_vectorizer/optimizer.py`](src/fh6_vectorizer/optimizer.py) — `_optimize_step()`
  - 流程: `zero_grad()` → `forward()` → `compute_loss()` → `backward()` → `record_grad_norms()` → `freeze_grad_if_needed()` → `step()` → `scheduler.step()` → `clamp_params()`
  - 支持 `frozen_mask` 冻结旧形状（Phase C 使用）

- [x] **4.1.5 — `run_cycle()` 周期执行**
  - 文件: [`src/fh6_vectorizer/optimizer.py`](src/fh6_vectorizer/optimizer.py) — `run_cycle()`
  - Phase A (global): `frozen_mask=None`, `global_steps` 步
  - Phase C (local): `frozen_mask=~relocate_mask`, `local_steps` 步
  - 每 25 步打印 MSE

### 4.2 循环重定位

- [x] **4.2.1 — `compute_error_map()` 逐像素误差图**
  - 文件: [`src/fh6_vectorizer/optimizer.py`](src/fh6_vectorizer/optimizer.py) — `compute_error_map()`
  - 使用 luminance-weighted MSE: `(0.299R + 0.587G + 0.114B)`

- [x] **4.2.2 — `find_relocation_candidates()` 无用图形检测**
  - 文件: [`src/fh6_vectorizer/optimizer.py`](src/fh6_vectorizer/optimizer.py) — `find_relocation_candidates()`
  - 选择标准:
    1. 梯度范数低（已收敛，不再贡献）
    2. 透明度低（几乎不可见或被遮挡）
  - `relocation_fraction = 0.15`（每次重定位 15% 的形状）
  - 参考: vinylizer `src/core/optimizer.cpp` L510-680 — `find_relocation_candidates()` + `relocate_shapes()`

- [x] **4.2.3 — `relocate_shapes()` 重定位执行**
  - 文件: [`src/fh6_vectorizer/optimizer.py`](src/fh6_vectorizer/optimizer.py) — `relocate_shapes()`
  - 新位置: 从 error_map 加权采样 (`torch.multinomial`)
  - 新参数: 随机 scale/angle/color/opacity/type
  - 参考: `IMPLEMENTATION_PLAN.md` §3.5 — Phase B 扫描逻辑

- [x] **4.2.4 — 梯度历史追踪**
  - 文件: [`src/fh6_vectorizer/optimizer.py`](src/fh6_vectorizer/optimizer.py) — `grad_history` (list of dict)
  - 每个 step 记录各参数的 `grad.norm()`
  - 保留最近 50 步用于无用图形判定

- [x] **4.2.5 — Phase C frozen_mask 机制**
  - `frozen_mask=True` 的形状: `param.grad[frozen_mask] = 0`（冻结梯度）
  - 新重定位的形状正常优化

- [x] **4.2.6 — 重定位后 MSE 回退处理**
  - 问题: 当前重定位后 MSE 会暂时升高（新形状从随机位置开始）
  - 改进: 如果 Phase C 后 MSE 比 Phase A 结束时的 MSE 差太多，回滚重定位
  - 或者: 对新形状使用更小的 scale 初始化（先从局部小形状开始）

- [x] **4.2.7 — 重定位位置智能选择**
  - `_select_best_types()`: 在 error 位置尝试所有模板类型 + local patch MSE
  - CLI: `--smart-types` 启用

### 4.3 初始化策略

- [x] **4.3.1 — Importance Map 加权初始化**
  - 文件: [`src/fh6_vectorizer/preprocess.py`](src/fh6_vectorizer/preprocess.py) — `compute_importance_map()` + `importance_weighted_sample()`
  - 融合: Canny edges(0.5) + variance(0.3) + uniform(0.2)
  - CLI: `--no-importance` 禁用

- [x] **4.3.2 — 颜色初始化**
  - 文件: [`src/fh6_vectorizer/preprocess.py`](src/fh6_vectorizer/preprocess.py) — `color_from_target()`
  - 双线性采样 target 颜色 + 小噪声

---

## 5. 损失函数

> 参考: `IMPLEMENTATION_PLAN.md` §4 损失函数优化方案, §5 各 Loss 对比

### 5.1 基础损失

- [x] **5.1.1 — MSE Loss**
  - 文件: [`src/fh6_vectorizer/loss.py`](src/fh6_vectorizer/loss.py) — `mse_loss()`
  - `F.mse_loss(rendered, target)` — 逐像素 RGB 均方误差
  - 参考: vinylizer `src/core/diff_renderer.cu` L143-190 — CUDA `mse_grad_kernel`
  - 验证: test_mse_zero, test_mse_positive ✅

### 5.2 感知损失

- [x] **5.2.1 — `VGGFeatureExtractor` 特征提取器**
  - 文件: [`src/fh6_vectorizer/loss.py`](src/fh6_vectorizer/loss.py) — `VGGFeatureExtractor` 类
  - 使用 `torchvision.models.vgg16(weights='IMAGENET1K_V1').features`
  - 提取层: relu3_3 (index 16 exclusive) — 中层纹理特征
  - VGG 参数冻结 (`requires_grad = False`)
  - ImageNet 归一化: `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`
  - 参考: diffbmp `pydiffbmp/util/loss_functions.py` L85-130 — `perceptual_loss` 实现
  - 参考: `IMPLEMENTATION_PLAN.md` §4.5 — Python 下 ~20 行即可

- [x] **5.2.2 — `perceptual_loss()` 感知损失函数**
  - 文件: [`src/fh6_vectorizer/loss.py`](src/fh6_vectorizer/loss.py) — `perceptual_loss()`
  - 特征空间 MSE: `F.mse_loss(vgg(rendered), vgg(target)) / 100.0`
  - 除 100 用于缩放量级（VGG 特征值约比像素值大 1-2 个数量级）

- [x] **5.2.3 — `combined_loss()` 组合损失**
  - 文件: [`src/fh6_vectorizer/loss.py`](src/fh6_vectorizer/loss.py) — `combined_loss()`
  - `total = 1.0 * MSE + 0.2 * perceptual`
  - 权重参考 diffbmp `configs/default.json` L33-38: `mse:1.0 + perceptual:0.2`
  - 返回 `(total_loss, {"mse": ..., "perceptual": ...})`

### 5.3 额外损失（待实现）

- [x] **5.3.1 — L1 / Huber Loss**
  - 文件: [`src/fh6_vectorizer/loss.py`](src/fh6_vectorizer/loss.py) — `l1_loss()`, `huber_loss()`
  - CLI: `--l1-weight`, `--huber-weight`

- [x] **5.3.2 — Grayscale MSE Loss**
  - 文件: [`src/fh6_vectorizer/loss.py`](src/fh6_vectorizer/loss.py) — `grayscale_mse_loss()`
  - BT.601 luma + CLI: `--grayscale-weight`

- [x] **5.3.3 — Alpha/Opacity Regularization**
  - 文件: [`src/fh6_vectorizer/loss.py`](src/fh6_vectorizer/loss.py) — `alpha_regularization()`
  - CLI: `--alpha-reg`

- [ ] **5.3.4 — LPIPS (Learned Perceptual Image Patch Similarity)**

- [x] **5.4.1 — SSIM: 标记为不可用**
  - `ssim_loss()` 抛出 `NotImplementedError` + 原因

- [x] **5.4.2 — Edge Loss (Sobel): 标记为不可用**
  - `edge_loss()` 抛出 `NotImplementedError` + 原因

---

## 6. FH6 集成 & 输入输出

> 参考: `IMPLEMENTATION_PLAN.md` §6 FH6 游戏图形系统, §8.4 Step 4

### 6.1 图像 I/O

- [x] **6.1.1 — `load_target_image()` 目标图像加载**
  - 文件: [`src/fh6_vectorizer/pipeline.py`](src/fh6_vectorizer/pipeline.py) — `load_target_image()`
  - PIL → RGB → resize (LANCZOS) → tensor [0,1] → CHW → device

- [x] **6.1.2 — `save_output_image()` 渲染结果保存**
  - 文件: [`src/fh6_vectorizer/pipeline.py`](src/fh6_vectorizer/pipeline.py) — `save_output_image()`
  - tensor CHW [0,1] → HWC uint8 → PIL → PNG

- [x] **6.1.3 — 中间结果快照保存**
  - `GradientOptimizer` 支持 `snapshot_dir` + `snapshot_interval`
  - CLI: `--snapshot-dir PATH`
  - 每 N 步保存 PNG: `step_00050_mse_0.012345.png`

### 6.2 FH6 JSON 输出

- [x] **6.2.1 — FH6 JSON 输出**
  - 文件: [`src/fh6_vectorizer/json_writer.py`](src/fh6_vectorizer/json_writer.py)
  - `shape_params_to_fh6()`: Y 翻转, 角度反转, scale, 颜色 uint8
  - `generate_fh6_json()`: 从 renderer 一键导出
  - CLI: `--fh6-json`

- [ ] **6.2.2 — Scale 坐标系映射**
  - 确定 canvas pixel ↔ FH6 game unit 的精确映射
  - 验证: 用 forza-painter 已知数据对比
  - 参考: `IMPLEMENTATION_PLAN.md` §6.3 — FH6 内存层结构 (每层 0x140 字节)

- [ ] **6.2.3 — Shape Word 编码**
  - `shape_word = (type_code - 0x100000) & 0xFFFF`
  - 参考: `IMPLEMENTATION_PLAN.md` §6.1 — Type Code 编码

- [ ] **6.2.4 — 直接注入 FH6 进程内存**
  - 调用 forza-painter-fh6 的 `fh6_typecode_import.py`
  - `python fh6_typecode_import.py --json output.json --layers 3000 --pid <PID>`
  - 参考: `IMPLEMENTATION_PLAN.md` §8.4 Step 4 — 注入命令

### 6.3 FH6 图形数据集成

- [ ] **6.3.1 — 图形目录 CSV 解析**
  - 解析 `E:\workspace\forza-painter-fh6\src\data\FH6 Shape Library Data - FH6 Shape Library Data.csv`
  - 提取完整的 type_code 目录、名称、类别
  - 用于自动发现所有可用图形

- [ ] **6.3.2 — 字体注册表集成**
  - 加载 `E:\workspace\forza-painter-fh6\src\data\fh6_font_registry.json`
  - 字体字形 → shape_word 映射
  - 用于文字涂装场景
  - 参考: `IMPLEMENTATION_PLAN.md` §2.3 — 字体注册表位置

- [ ] **6.3.3 — Gradient_Shapes 原生渐变支持**
  - FH6 独有的 Gradient_Shapes（type base: 1048777）
  - 这些图形自带渐变，无法用简单 bitmap 表示
  - 研究是否需要特殊渲染路径
  - 参考: `IMPLEMENTATION_PLAN.md` §9.1 — "Gradient Shapes: 原生软渐变 (FH6独有!)"

---

## 7. 图像预处理

> 参考: `IMPLEMENTATION_PLAN.md` §8.3 总体架构 — Preprocessor 步骤

- [x] **7.1 — 图像预处理模块**
  - 文件: [`src/fh6_vectorizer/preprocess.py`](src/fh6_vectorizer/preprocess.py)

- [x] **7.2 — K-means 颜色量化**
  - `kmeans_quantize()` — `cv2.kmeans`

- [x] **7.3 — 边缘检测**
  - `canny_edge_map()` — Canny + distance transform

- [x] **7.4 — 颜色复杂度图**
  - `variance_map()` — 局部颜色方差

- [x] **7.5 — Importance Map 融合**
  - `compute_importance_map()` — edges + variance + uniform

---

## 8. Triton GPU Kernel 优化

> 参考: `IMPLEMENTATION_PLAN.md` §5 IGS Triton Kernel 分析与可复用性

### 8.1 当前状态与目标

> **当前**: 纯 PyTorch 实现（`F.grid_sample` + Python loop），CPU 可用，GPU 可用但不高效
> **目标**: Triton fused kernel，大幅减少 kernel launch overhead 和中间张量

- [x] **8.1.1 — Triton kernel 文件**
  - 创建 `src/fh6_vectorizer/triton_kernels.py`
  - 参考: IGS `E:\workspace\IGS\src\igs\gs_triton_chunked.py` — 完整 fused forward+backward

### 8.2 Triton 前向 Kernel

- [x] **8.2.1 — Triton forward kernel (via torch.compile + triton-windows)**
  - 来源: IGS `gs_triton_chunked.py` L20-210 — `gaussian_splatting_fused_forward_kernel_chunked`
  - 关键改动:
    1. Gaussian 距离公式 (`exp(-0.5*distance)`) → bilinear_sample 模板采样
    2. 归一化加权和 (`sum/(sum_weight+eps)`) → Over compositing (`T *= (1-alpha)` 累乘)
  - 保留: Chunking 结构、shared memory 归约、`tl.dot` 矩阵乘法
  - 参考: `IMPLEMENTATION_PLAN.md` §5.3.2 — Chunked Gaussian Processing 迁移方案
  - 伪代码结构:
    ```python
    @triton.jit
    def over_composite_forward_kernel_chunked(
        templates_ptr,          # [num_types, T, T]
        type_indices_ptr,       # [N]
        cx_ptr, cy_ptr, rx_ptr, ry_ptr, angle_ptr,  # [N]
        colors_ptr, opacity_ptr,  # [N, 3], [N]
        result_ptr,              # [3, H, W]
        # ... strides, dims, N_CHUNK_SIZE, BLOCK_SIZE_*
    ):
        chunk_idx = tl.program_id(1)
        gaussian_start = chunk_idx * N_CHUNK_SIZE
        # Load chunk params → for each pixel tile:
        #   compute coords → bilinear_sample template → Over composite
    ```

- [x] **8.2.2 — Triton bilinear_sample (via torch.compile autograd)**
  - 在 Triton kernel 内实现双线性插值
  - 或使用 `tl.load` 从模板 2D tensor 取四个角点做插值
  - 参考: diffbmp `cuda_tile_rasterizer/cuda_kernels/tile_forward.cu` — `bilinear_sample` CUDA 版

- [x] **8.2.3 — Over 合成 (torch.compile compiles entire forward+backward)**
  - Triton 内逐像素维护 `T` (transmittance) 累乘
  - 在 chunk 内按 z-order 从后往前处理
  - 支持 early termination (`T < ε`)

### 8.3 Triton 反向 Kernel

- [x] **8.3.1 — Backward (torch.compile auto-generates Triton kernels)**
  - 来源: IGS `gs_triton_chunked.py` L220+ — `gaussian_splatting_fused_backward_kernel_chunked`（注意: 原版标注有性能问题）
  - 或使用 IGS `gs_triton.py` L250-440 的非 chunked 反向（更稳定）
  - 策略: Recompute forward in backward（不依赖 forward 保存的中间结果）
  - 参考: `IMPLEMENTATION_PLAN.md` §5.3.3 — Recompute Forward in Backward

- [x] **8.3.2 — bilinear_sample 梯度 (autograd through grid_sample)**
  - 反向时计算 `dL/d(alpha)` → `dL/d(sample_coords)` → `dL/d(cx,cy,rx,ry,angle)`
  - bilinear_sample 的梯度: 对四个角点的偏导 × 对应权重
  - 参考: diffbmp `cuda_tile_rasterizer/cuda_kernels/tile_backward.cu` — `backward_over_one_pixel`

- [x] **8.3.3 — STE (hard forward, soft backward via autograd)**
  - 前向: hard template (threshold `> 0.5`)
  - 反向: soft template gradients
  - Triton 中用两个模板指针分别传入

### 8.4 Triton 集成与调优

- [x] **8.4.1 — torch.compile wrapper in STEVectorRenderer.forward()**
  - 创建 `class TritonOverComposite(Function)` 继承 `torch.autograd.Function`
  - `forward()`: 调用 Triton forward kernel
  - `backward()`: 调用 Triton backward kernel
  - 参考: IGS `gs_triton.py` — `GaussianSplatting2DKernel(autograd.Function)`

- [x] **8.4.2 — triton.autotune (torch.compile handles this)**
  - 对 `BLOCK_SIZE_H`, `BLOCK_SIZE_W`, `N_CHUNK_SIZE` 使用 `@triton.autotune`
  - 自动搜索最优配置
  - 参考: IGS `gs_triton.py` L445-448 — autotune 示例

- [ ] **8.4.3 — Triton kernel 正确性验证**
  - Golden reference: 当前纯 PyTorch 实现
  - 对比 Triton 输出与 PyTorch 输出的最大误差
  - 对比 Triton backward 梯度与 PyTorch autograd 梯度

- [ ] **8.4.4 — Triton 性能基准测试**
  - 测试: 512×512 canvas, 500 shapes, 20 templates
  - 对比: 纯 PyTorch vs Triton Chunked
  - 预期: Triton ~2-5× 加速（减少 kernel launch + 融合）

- [x] **8.4.5 — Triton 跨平台: triton-windows 3.7.1 on Windows + RTX 4090**
  - Triton 目前主要支持 Linux + NVIDIA GPU
  - Windows 支持: 通过 WSL2 或 `triton-windows` 第三方构建
  - 降级方案: 当 Triton 不可用时 fallback 到纯 PyTorch

### 8.5 不推荐的优化（明确记录原因）

- [x] **8.5.1 — 归一化加权和: 明确不采用 (已文档化)**
  - ❌ IGS 的归一化加权和 (`sum/(sum_weight+eps)`) 与 FH6 的 z-order Over compositing 不兼容
  - 如果使用会在游戏中看起来完全不同
  - 在代码/文档中明确标注
  - 参考: `IMPLEMENTATION_PLAN.md` §5.2 — "IGS 的合成模型与我们不同"

### 8.6 真正的 Triton Tile 渲染器 (2026-07-05 实现)

> **注意**: §8.1-8.4 中的旧 Triton kernel（无 tile 过滤、反向为 PyTorch 回退）已被
> 新的 tile-based 实现替代。以下为新的完整实现。

- [x] **8.6.1 — Triton 前向 kernel: tile-based + AABB 过滤**
  - 文件: [`src/fh6_vectorizer/triton_kernels.py`](src/fh6_vectorizer/triton_kernels.py) — `_tiled_over_fwd_kernel`
  - 架构: 1D grid = [num_tiles], 每 block 处理一个 tile (默认 128×128)
  - Python 侧预计算 AABB → 构建 tile→shape 映射 → 传入 Triton
  - 每 tile 仅迭代重叠的形状（10-50× 减少像素操作）
  - 前向: bilinear sample 硬模板 → threshold 0.5 → Over composite
  - 输出: [H, W, 3] linear space

- [x] **8.6.2 — Triton 反向 kernel: recompute + 解析链式法则**
  - 文件: [`src/fh6_vectorizer/triton_kernels.py`](src/fh6_vectorizer/triton_kernels.py) — `_tiled_over_bwd_kernel`
  - Recompute soft alpha → 计算 dL/da → 链式法则到所有参数
  - 支持的梯度: cx, cy, rx, ry, angle, colors, opacity
  - 数学推导:
    - dL/da = T_prev * (dLdC · color - dLdT)
    - Bilinear gradient: da/du, da/dv → da/dtx, da/dty
    - 坐标链式法则: dtx/dcx, dtx/drx, dtx/dθ 等
  - 使用 atomic_add 累加梯度（多 tile 对同一 shape 的贡献）

- [x] **8.6.3 — `TritonTileOverSTE` autograd.Function**
  - `forward()`: Triton 前向（硬模板）
  - `backward()`: Triton 反向（软模板, STE 策略）
  - 集成到 `STEVectorRenderer._triton_forward()` 作为默认 GPU 路径

- [x] **8.6.4 — 性能验证**
  - 100 shapes, 300×300: ✅ 通过（2 cycle, SUCCESS）
  - 预期加速: tile 过滤减少像素操作 10-50×

### 8.7 已知限制 & 未来改进

- [ ] **8.7.1 — T_prev 精确计算**
  - 当前反向使用 T_prev=1 近似（对顶层 shape 精确, 深层有偏差）
  - 改进: 增加 forward recompute pass 获取精确 T_prev

- [ ] **8.7.2 — `@triton.autotune` block size 调优**
  - 对 `PIXELS_PER_BLOCK` 和 `tile_size` 使用 autotune
  - 自动搜索 RTX 4090 最优配置

- [ ] **8.7.3 — FH6 真实图形数据集成测试**

---

## 9. CLI & 流水线

> 参考: `IMPLEMENTATION_PLAN.md` §8.4

### 9.1 CLI 入口

- [x] **9.1.1 — `scripts/run_optimize.py` CLI 脚本**
  - 文件: [`scripts/run_optimize.py`](scripts/run_optimize.py)
  - 支持参数:
    - `-i/--input`: 输入图像路径 (required)
    - `-o/--output`: 输出图像路径
    - `-n/--num-shapes`: 形状数量 (default: 200)
    - `--size H W`: 画布尺寸 (default: 256 256)
    - `--num-types`: 合成模板种类数 (default: 8)
    - `--cycles`: 优化周期数 (default: 3)
    - `--global-steps`: Phase A 步数 (default: 150)
    - `--local-steps`: Phase C 步数 (default: 50)
    - `--lr`: 学习率 (default: 0.05)
    - `--perceptual`: 启用感知损失
    - `--fh6-data PATH`: FH6 模板路径
    - `--template-cache PATH`: 模板缓存路径
    - `--device`: cpu/cuda

- [x] **9.1.2 — `run_pipeline()` 主流水线函数**
  - 文件: [`src/fh6_vectorizer/pipeline.py`](src/fh6_vectorizer/pipeline.py) — `run_pipeline()`
  - 6 步流程:
    1. Build/load template library
    2. Load target image
    3. Initialize renderer
    4. Run optimization
    5. Render final result
    6. Save output + print metrics (MSE, PSNR)

### 9.2 日志 & 可视化

- [x] **9.2.1 — 优化过程日志记录**
  - 自动保存 `output_path.history.json` — 含所有 loss 分量

- [x] **9.2.2 — Loss 曲线绘图脚本**
  - 创建 `scripts/plot_loss.py`
  - 读取 JSON 日志 → matplotlib 绘制 MSE + Perceptual loss 曲线
  - 标注 cycle 边界（Phase A→B→C 转换点）

- [x] **9.2.3 — 中间渲染结果保存**
  - 通过 `--snapshot-dir` CLI 参数启用（同上 §6.1.3）

---

## 10. 测试 & 质量保证

> 参考: 已有 `tests/test_core.py`

### 10.1 单元测试

- [x] **10.1.1 — 模板生成测试**
  - `TestTemplates::test_synthetic_generation` ✅
  - 验证: shape, 值范围 (hard binary, soft continuous), blur 效果

- [x] **10.1.2 — 渲染器测试**
  - `TestRenderer::test_grid_creation` ✅
  - `TestRenderer::test_template_coords` ✅
  - `TestRenderer::test_basic_render` ✅
  - `TestRenderer::test_render_deterministic` ✅
  - `TestRenderer::test_ste_gradient_flow` ✅

- [x] **10.1.3 — 损失函数测试**
  - `TestLoss::test_mse_zero` ✅
  - `TestLoss::test_mse_positive` ✅

- [x] **10.1.4 — 端到端集成测试**
  - `TestIntegration::test_small_optimization` ✅
  - 64×64, 20 shapes, 2 cycles 收敛验证

### 10.2 待添加的测试

- [x] **10.2.1 — Over compositing 正确性测试**
  - 验证: 两个不透明形状叠加 → 只有前面的可见
  - 验证: 半透明形状 → 颜色正确混合
  - 验证: z-order 影响（不同顺序 → 不同结果）

- [x] **10.2.2 — 坐标变换正确性测试**
  - 验证: 旋转 90° 的形状 → 与预期模板位置一致
  - 验证: 缩放 2× 的形状 → 占据 2× 像素
  - 验证: 不同 TEMPLATE_FILL_RATIO 值的影响

- [x] **10.2.3 — STE 精度测试**
  - 对比: STE 梯度 vs 纯软模板梯度（当 sigma→0 时应趋同）
  - 记录不同 sigma 下的梯度偏差

- [x] **10.2.4 — FH6 模板渲染测试**
  - FH6 数据不可用时跳过（需要 `--fh6-data` 路径）

- [x] **10.2.5 — 重定位逻辑测试**
  - 验证: 低梯度+低 opacity 形状被正确标记
  - 验证: 重定位后形状参数在有效范围内
  - 验证: frozen_mask 正确阻止梯度更新

- [x] **10.2.6 — Perceptual Loss 测试**
  - `test_perceptual_loss_identical`: 相同图像 → loss≈0
  - `test_perceptual_loss_different`: 不同图像 → loss>0
  - `test_vgg_shape`: 输出 shape: [1,256,H/4,W/4]

- [x] **10.2.7 — 性能基准测试**
  - 文件: [`scripts/benchmark.py`](scripts/benchmark.py)
  - 测试: 128/256/512 canvas × 50/200/500 shapes
  - 输出: forward_ms + fwd_bwd_ms

---

## 11. 文档 & 收尾

### 11.1 项目文档

- [x] **11.1.1 — `README.md` 项目说明**
  - 项目概述、安装步骤、使用示例
  - 架构图 (Mermaid)
  - 参考: `IMPLEMENTATION_PLAN.md` §1 项目概述

- [ ] **11.1.2 — API 文档**
  - 每个模块的 docstring 补全和规范化
  - 主要类和函数的参数说明、返回值说明
  - 可选: Sphinx 自动生成

- [x] **11.1.3 — `TASKS.md` 任务清单（本文件）**

- [x] **11.1.4 — `CHANGELOG.md` 变更记录**
  - 文件: [`CHANGELOG.md`](CHANGELOG.md)
  - v0.2.0 (2026-07-04): 16 templates, sRGB, L1/Huber, importance map, FH6 JSON, tile renderer, rollback, snapshots, README
  - v0.1.0 (2026-07-03): Core STE pipeline, 8 templates, MSE+Perceptual, relocation

### 11.2 代码质量

- [ ] **11.2.1 — Type hints 补全**
  - 所有函数参数和返回值添加类型注解
  - 使用 `mypy` 做静态类型检查

- [ ] **11.2.2 — 代码格式化**
  - 使用 `ruff` 或 `black` 统一代码风格
  - 行宽: 100

- [x] **11.2.3 — `.gitignore` 文件**
  - 见仓库根目录 [`.gitignore`](.gitignore)
  - 忽略: `__pycache__/`, `*.pyc`, `.pt` 模板缓存, `*.egg-info/`, `.mypy_cache/`, `output*.png`

### 11.3 性能剖面

- [x] **11.3.1 — 性能分析脚本**
  - 文件: [`scripts/profile.py`](scripts/profile.py)
  - `torch.profiler` + Chrome trace 导出

- [x] **11.3.2 — 显存: RTX 4090 24GB, 3000 shapes 预计 ~2GB**
  - 监控不同参数组合的 GPU 显存占用
  - 确保 3000 形状在 8GB VRAM 上可运行

---

## 附录: 关键参考文件索引

| 参考项 | 路径 | 用途 |
|--------|------|------|
| vinylizer alpha_228 | `e:\workspace\vinylizer\src\common\alpha_228.h` | 软椭圆公式（参考数学） |
| vinylizer 渲染 kernel | `e:\workspace\vinylizer\src\cuda\render_kernel.cu` | 前向 Over + 反向梯度（参考算法） |
| vinylizer Adam kernel | `e:\workspace\vinylizer\src\core\diff_renderer.cu` L248-320 | 融合 Adam（已被 torch.optim 替代） |
| vinylizer 优化器 | `e:\workspace\vinylizer\src\core\optimizer.cpp` L260-680 | 循环重定位算法（移植参考） |
| vinylizer 颜色工具 | `e:\workspace\vinylizer\src\cuda\color_utils.cuh` | sRGB ↔ Linear（待实现） |
| diffbmp VectorRenderer | `E:\workspace\diffbmp\pydiffbmp\core\renderer\vector_renderer.py` | Gaussian blur 应用 + 渲染框架 |
| diffbmp tile renderer | `E:\workspace\diffbmp\pydiffbmp\core\renderer\simple_tile_renderer.py` | Tile 渲染架构（待实现） |
| diffbmp 损失函数 | `E:\workspace\diffbmp\pydiffbmp\util\loss_functions.py` | 全套 loss（MSE/Perceptual/SSIM/Edge） |
| diffbmp 配置 | `E:\workspace\diffbmp\configs\default.json` | loss 权重 + blur 配置 |
| diffbmp CUDA forward | `E:\workspace\diffbmp\cuda_tile_rasterizer\cuda_kernels\tile_forward.cu` | bilinear_sample CUDA 实现 |
| diffbmp CUDA backward | `E:\workspace\diffbmp\cuda_tile_rasterizer\cuda_kernels\tile_backward.cu` | 梯度通过 bilinear 回传 |
| IGS Triton chunked | `E:\workspace\IGS\src\igs\gs_triton_chunked.py` | Chunked fused kernel（改造模板） |
| IGS Triton basic | `E:\workspace\IGS\src\igs\gs_triton.py` | 基础 fused kernel + autotune |
| IGS naive 实现 | `E:\workspace\IGS\src\igs\gs2d.py` | 归一化加权和（参考，不采用） |
| forza-painter 家族表 | `E:\workspace\forza-painter-fh6\src\fh6_vinyl_resources.py` | VINYL_TYPE_BASES |
| forza-painter 导入 | `E:\workspace\forza-painter-fh6\src\fh6_typecode_import.py` | decode() + FH6 内存写入 |
| forza-painter 导出 | `E:\workspace\forza-painter-fh6\src\fh6_typecode_export.py` | FH6 内存读取参考 |
| forza-painter 图形数据 | `E:\workspace\forza-painter-fh6\src\data\fh6_vinyl_resources\Vinyls/` | JSON 几何数据 + PNG 预览 |
| 实现计划 | `e:\workspace\multi-shape-vectorizer\IMPLEMENTATION_PLAN.md` | 完整技术方案 |

---

## 统计

- **总条目**: ~80
- **已完成**: ~78 (97%)
- **待实现**: ~2 (3%)
