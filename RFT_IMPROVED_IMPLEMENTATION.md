
# 改进版 RFT 实现 - 基于 Terradynamics 论文

## 概述

本版本实现了基于 "Terradynamics of Legged Locomotion on Granular Media" 论文的改进版 RFT（Resistive Force Theory），
核心改进是**首次正确使用了 `rft_generic_M` 矩阵**来计算足端在沙中的受力。

## 文件结构

### 新增文件（改进版）

| 文件路径 | 说明 |
|---------|------|
| `sand_env_improved_cfg.py` | 改进版沙地环境配置（保持材质不变） |
| `hexapod_sand_play_env_improved.py` | 改进版 Play 环境（使用 RFT 矩阵） |
| `hexapod_sand_train_env_improved.py` | 改进版 Train 环境（使用 RFT 矩阵） |

### 注册的新环境 ID

| 环境 ID | 说明 |
|---------|------|
| `Isaac-Velocity-Sand-Improved-Hexapod-v0` | 改进版训练环境 |
| `Isaac-Velocity-Sand-Improved-Hexapod-Play-v0` | 改进版演示环境 |

## 核心改进

### 1. 正确使用 `rft_generic_M` 矩阵

`rft_generic_M = [0.206, 0.169, 0.212, 0.358, 0.055, -0.124, 0.253, 0.007, 0.088]`

矩阵参数含义：
| 索引 | 符号 | 物理含义 |
|------|------|---------|
| 0 | Fz0 | Z 方向深度依赖系数 |
| 1 | Fx0 | X 方向深度依赖系数 |
| 2 | Fy0 | Y 方向深度依赖系数 |
| 3 | Fzv | Z 方向速度依赖系数 |
| 4 | Fxv | X 方向速度依赖系数 |
| 5 | Fyv | Y 方向速度依赖系数 |
| 6 | Fzα | Z 方向角度依赖系数 |
| 7 | Fxα | X 方向角度依赖系数 |
| 8 | Fyα | Y 方向角度依赖系数 |

### 2. 力计算模型

#### Z 方向力（垂直）
```
Fz = ζ*A*(Fz0*depth + Fzv*downward_speed + Fzα*angle_factor*depth) 
    - ζ*A*Fzv*0.2*upward_speed
```

#### X 方向力（水平前向）
```
Fx = -ζ*A*(Fx0*depth*sign(vx) + Fxv*vx + Fxα*angle_factor*vx)
```

#### Y 方向力（水平侧向）
```
Fy = -ζ*A*(Fy0*depth*sign(vy) + Fyv*vy + Fyα*angle_factor*vy)
```

### 3. 考虑足端姿态

新增 `_quat_to_rotation_matrix` 方法，计算足端法向量，用于：
- 角度因子计算 `angle_factor = 1 - |cos(angle)|`
- 角度越大，角度依赖项的影响越大

## 与原实现的对比

| 特性 | 原实现 | 改进版实现 |
|------|--------|-----------|
| `rft_generic_M` 使用 | ❌ 未使用 | ✅ 核心参数 |
| 力模型 | 简单粘性阻尼 | 基于论文的 RFT |
| 角度依赖 | ❌ 忽略 | ✅ 考虑足端姿态 |
| 分别计算 X/Y/Z 力 | ❌ X/Y 合并 | ✅ 分别计算 |
| 垂直力计算 | 简单线性 | 深度 + 速度 + 角度 |

## 使用方法

### 测试改进版环境

```python
# 在 zero_agent1.py 或类似脚本中使用
env_id = "Isaac-Velocity-Sand-Improved-Hexapod-Play-v0"
```

### 训练改进版策略

```python
env_id = "Isaac-Velocity-Sand-Improved-Hexapod-v0"
```

## 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `sand_zeta` | 0.5 | 介质阻力系数 |
| `foot_area` | 0.01 | 足端面积（平方米） |
| `foot_bottom_offset` | 0.06 | 足底相对于足端 body 原点的偏移 |
| `min_sink_depth` | 0.001 | 最小入沙深度 |
| `contact_force_threshold` | 1.0 | 判断足端在沙中的接触力阈值 |

## 下一步工作建议

1. **参数校准**：如果有实验数据，可校准 `sand_zeta` 和力计算公式中的系数
2. **完整面元分解**：将足端分解为小面元，对每个面元应用 RFT 后积分
3. **足端真实几何**：从 URDF/OBJ 中获取真实足端面积和形状

## 相关文件

- 原环境：`sand_env_cfg.py`, `hexapod_sand_play_env.py`, `hexapod_sand_train_env.py`
- 分析文档：`RFT_TERRADYNAMICS_ANALYSIS.md`
