
# 改进版 RFT 环境 - 使用指南

## 概述

本文档说明如何使用改进版的沙地环境（`zero_agent3.py`），该环境正确实现了基于 "Terradynamics of Legged Locomotion on Granular Media" 论文的 RFT 力计算，
**首次正确使用了 `rft_generic_M` 矩阵**！

## 📁 相关文件

| 文件 | 作用 |
|------|------|
| [zero_agent3.py](file:///home/ubuntu/IsaacLab-LiBRHexapod/scripts/environments/zero_agent3.py) | 改进版 RFT 环境的运行脚本 |
| [hexapod_sand_play_env_improved.py](file:///home/ubuntu/IsaacLab-LiBRHexapod/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/hexapod/hexapod_sand_play_env_improved.py) | 改进版 Play 环境实现 |
| [hexapod_sand_train_env_improved.py](file:///home/ubuntu/IsaacLab-LiBRHexapod/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/hexapod/hexapod_sand_train_env_improved.py) | 改进版 Train 环境实现 |
| [sand_env_improved_cfg.py](file:///home/ubuntu/IsaacLab-LiBRHexapod/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/hexapod/sand_env_improved_cfg.py) | 改进版环境配置 |
| [__init__.py](file:///home/ubuntu/IsaacLab-LiBRHexapod/source/isaaclab_tasks/manager_based/locomotion/velocity/config/hexapod/__init__.py) | 环境注册（已更新） |

## 🎮 环境 ID

| ID | 用途 |
|----|------|
| `Isaac-Velocity-Sand-Improved-Hexapod-Play-v0` | 演示/测试（录制视频） |
| `Isaac-Velocity-Sand-Improved-Hexapod-v0` | 训练 |

## 🚀 快速开始

### 1. 运行改进版环境（录制视频）

```bash
cd /home/ubuntu/IsaacLab-LiBRHexapod
python scripts/environments/zero_agent3.py
```

### 2. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--task` | `Isaac-Velocity-Sand-Improved-Hexapod-Play-v0` | 任务名称 |
| `--num_envs` | None | 环境数量 |
| `--disable_fabric` | False | 禁用 fabric |

### 3. 输出内容

运行后会在 `my_videos/` 目录下生成视频文件：
- `hexapod_sand_improved_demo-episode-0.mp4`

同时会在控制台输出移动距离统计：
```
============================================================
改进版沙地环境（zero_agent3）移动统计
============================================================
总模拟时间：300步 = 6.0秒
初始位置：x=0.000m, y=0.000m
最终位置：x=1.234m, y=0.100m
x方向移动距离：1.234米
二维平面总移动距离：1.238米
相当于移动了：3.5个体长（体长≈0.35米）
============================================================
```

## 🔬 原环境 vs 改进版对比

### 运行原环境（zero_agent1）

```bash
python scripts/environments/zero_agent1.py
```
- 环境：`Isaac-Velocity-Sand-Hexapod-Play-v0`
- RFT：简单粘性阻尼，`rft_generic_M` 未使用

### 运行改进版环境（zero_agent3）

```bash
python scripts/environments/zero_agent3.py
```
- 环境：`Isaac-Velocity-Sand-Improved-Hexapod-Play-v0`
- RFT：基于论文的完整实现，使用 `rft_generic_M` 矩阵，考虑角度依赖

## 📊 核心改进

### 原实现问题
- ❌ `rft_generic_M` 矩阵定义但完全未使用
- ❌ 力模型过于简化（仅粘性阻尼）
- ❌ 忽略足端姿态和角度

### 改进版优势
- ✅ **正确使用 `rft_generic_M` 矩阵**的9个系数
- ✅ 分别计算 X/Y/Z 三个方向的力
- ✅ 考虑足端法向量和角度依赖
- ✅ 力计算公式基于 Terradynamics 论文

## 🔧 调试输出

改进版环境会输出 RFT 调试信息（每50步一次）：
```
[DEBUG Improved RFT]: step=50 max_depth=0.004 max_fz=0.234 max_fx=0.056 max_fy=0.045 max_contact=2.345
```

## 📚 更多文档

- [RFT_TERRADYNAMICS_ANALYSIS.md](file:///home/ubuntu/IsaacLab-LiBRHexapod/RFT_TERRADYNAMICS_ANALYSIS.md) - 原实现与论文的详细对比分析
- [RFT_IMPROVED_IMPLEMENTATION.md](file:///home/ubuntu/IsaacLab-LiBRHexapod/RFT_IMPROVED_IMPLEMENTATION.md) - 改进版实现的详细技术说明

## 🧪 实验建议

可以对比运行 zero_agent1 和 zero_agent3，观察两种 RFT 实现的效果差异：
1. 先运行 `zero_agent1.py`（原环境）记录移动距离
2. 再运行 `zero_agent3.py`（改进版）记录移动距离
3. 对比两者的视频和移动统计，分析差异
