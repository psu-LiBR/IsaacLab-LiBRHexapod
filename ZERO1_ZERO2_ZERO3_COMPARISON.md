
# Zero1、Zero2、Zero3 环境对比分析

---

## 一、环境概述

| 脚本 | 环境 | 用途 |
|-----|-----|-----|
| [zero_agent2.py](file:///home/ubuntu/IsaacLab-LiBRHexapod/scripts/environments/zero_agent2.py) | 平地环境（Flat） | 对照组，机器人无额外沙地阻力 |
| [zero_agent1.py](file:///home/ubuntu/IsaacLab-LiBRHexapod/scripts/environments/zero_agent1.py) | 原始沙地环境 | 简化RFT阻力模型 |
| [zero_agent3.py](file:///home/ubuntu/IsaacLab-LiBRHexapod/scripts/environments/zero_agent3.py) | 改进沙地环境 | 基于论文思想的简化RFT模型 |

---

## 二、Zero2（平地）：无额外阻力

### 特点
- 使用 `Isaac-Velocity-Flat-Hexapod-Play-v0` 环境
- 物理引擎自带的地面接触
- **无任何代码添加的额外阻力或力**

### 物理原理
完全由 Isaac Sim/PhysX 的物理引擎处理：
- 正常的地面摩擦
- 腿部与地面的碰撞接触
- 无代码施加的外力

---

## 三、Zero1（原始沙地）：简化阻尼模型

### 环境代码
[hexapod_sand_play_env.py](file:///home/ubuntu/IsaacLab-LiBRHexapod/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/hexapod/hexapod_sand_play_env.py)

### 阻尼方式

#### 1. 垂直力（Z方向）
```python
vertical_force = self.sand_zeta * self.foot_area * (depth * 50.0 + downward_speed * 10.0)
vertical_force -= self.sand_zeta * self.foot_area * upward_speed * 2.0
```

**特点**：
- ✅ 考虑入沙深度（`depth`）
- ✅ 考虑向下/向上速度（`downward_speed` / `upward_speed`）
- ❌ **不使用 `rft_generic_M` 矩阵**
- ❌ **不考虑足端姿态角度**

#### 2. 水平力（XY方向）
```python
forces[..., 0:2] = -self.sand_zeta * 20.0 * foot_vel[..., 0:2]
```

**特点**：
- ✅ 纯速度比例阻尼（类似粘性摩擦）
- ✅ 与速度方向相反
- ❌ **不考虑入沙深度对水平力的影响**
- ❌ **不使用 `rft_generic_M` 矩阵**
- ❌ **不考虑足端姿态角度**

### 关键参数

| 参数 | Zero1值 | 物理含义 |
|-----|---------|---------|
| `sand_zeta` | 0.5 | 整体阻力系数 |
| `foot_area` | 0.01 m² | 足端面积 |
| `foot_bottom_offset` | 0.06 m | 脚底高度偏移 |
| `min_sink_depth` | 0.001 m | 最小入沙深度 |
| 垂直深度增益 | 50.0 (硬编码) | 深度对垂直力的影响 |
| 垂直速度增益 | 10.0 (硬编码) | 向下速度对垂直力的影响 |
| 向上卸载增益 | 2.0 (硬编码) | 向上速度的卸载系数 |
| 水平阻尼增益 | 20.0 (硬编码) | 水平速度的阻尼系数 |

### 理论缺陷
1. **完全忽略 `rft_generic_M` 矩阵**（虽然定义了但未用）
2. **力与深度/速度是简单线性关系**，而非论文中的非线性关系
3. **水平力与深度无关**，不符合真实沙地特性
4. **不考虑足端姿态**

---

## 四、Zero3（改进沙地）：基于论文思想的RFT

### 环境代码
[hexapod_sand_play_env_improved.py](file:///home/ubuntu/IsaacLab-LiBRHexapod/source/isaaclab_tasks/manager_based/locomotion/velocity/config/hexapod/hexapod_sand_play_env_improved.py)

### 阻尼方式

#### 1. 角度计算
```python
attack_angle = torch.acos(torch.clamp(torch.abs(foot_normal[..., 2]), 0.0, 1.0))
intrusion_angle = torch.atan2(-foot_vel[..., 2], tangential_speed + eps)
phase_angle = intrusion_angle - attack_angle
```

**特点**：
- ✅ 计算足端法向与地面的夹角（`attack_angle`）
- ✅ 计算速度方向与地面的夹角（`intrusion_angle`）
- ✅ 引入相位角（`phase_angle`）

#### 2. 使用 `rft_generic_M` 矩阵计算系数
```python
vertical_coeff = m[0] + m[1] * torch.cos(attack_angle) + m[2] * torch.sin(attack_angle)
drag_coeff = m[3] + m[4] * torch.cos(intrusion_angle) + m[5] * torch.sin(intrusion_angle)
coupling_coeff = m[6] + m[7] * torch.cos(phase_angle) + m[8] * torch.sin(phase_angle)
```

**特点**：
- ✅ **真正使用了 `rft_generic_M` 矩阵**
- ✅ 系数是角度的函数（余弦/正弦组合）
- ✅ 有垂直系数、阻力系数、耦合系数三种

#### 3. 垂直力
```python
vertical_force = self.sand_zeta * self.foot_area * (
    depth * self.vertical_depth_gain * vertical_coeff
    + downward_speed * self.vertical_speed_gain * (0.5 + coupling_coeff)
    - upward_speed * self.upward_relief_gain * drag_coeff
)
```

**特点**：
- ✅ 深度项 × `vertical_depth_gain` × 角度系数
- ✅ 向下速度项 × `vertical_speed_gain` × (0.5 + 耦合系数)
- ✅ 向上卸载项 × `upward_relief_gain` × 阻力系数

#### 4. 水平力
```python
horizontal_drag_mag = self.sand_zeta * self.foot_area * (
    depth * self.horizontal_depth_gain * drag_coeff
    + tangential_speed * self.horizontal_speed_gain * (0.5 + coupling_coeff)
)

posture_coupling = (
    self.sand_zeta * self.foot_area * depth.unsqueeze(-1)
    * self.posture_coupling_gain * coupling_coeff.unsqueeze(-1) * normal_xy
)

horizontal_force = -tangential_dir * horizontal_drag_mag.unsqueeze(-1) + posture_coupling
```

**特点**：
- ✅ **水平力与深度相关**（第一项）
- ✅ 包含速度阻尼项（第二项）
- ✅ 引入姿态耦合项（`posture_coupling`）
- ✅ 通过 `normal_xy`（足端法向在XY平面的投影）产生方向依赖

### 关键参数（最终调优值）

| 参数 | Zero3值 | 物理含义 | 与Zero1对比 |
|-----|---------|---------|-----------|
| `sand_zeta` | 3.5 | 整体阻力系数 | 0.5 → 3.5（增大7x） |
| `foot_area` | 0.02 m² | 足端面积 | 0.01 → 0.02（增大2x） |
| `foot_bottom_offset` | 0.0 m | 脚底高度偏移 | 0.06 → 0（修正） |
| `min_sink_depth` | 0.001 m | 最小入沙深度 | 不变 |
| `vertical_depth_gain` | 7000.0 | 垂直深度增益 | 50 → 7000（硬编码→参数） |
| `vertical_speed_gain` | 150.0 | 垂直速度增益 | 10 → 150（硬编码→参数） |
| `horizontal_depth_gain` | 5000.0 | 水平深度增益 | **新增** |
| `horizontal_speed_gain` | 100.0 | 水平速度增益 | **新增** |
| `posture_coupling_gain` | 0.2 | 姿态耦合增益 | **新增** |
| `upward_relief_gain` | 15.0 | 向上卸载增益 | 2 → 15（硬编码→参数） |
| `force_limit` | 40.0 N | 最大外力限制 | **新增** |

---

## 五、Zero1 与 Zero3 的理论对比

| 特性 | Zero1（原始沙地） | Zero3（改进沙地） |
|-----|------------------|------------------|
| **使用 rft_generic_M** | ❌ 定义但未用 | ✅ 核心部分 |
| **考虑足端姿态** | ❌ 完全忽略 | ✅ attack_angle / phase_angle |
| **水平力与深度关系** | ❌ 无关 | ✅ 深度项 × drag_coeff |
| **力的结构** | 简单线性阻尼 | 角度依赖系数 + 深度 + 速度 + 姿态耦合 |
| **垂直力模型** | 深度 + 速度 | 深度×垂直系数 + 速度×耦合系数 + 向上卸载×阻力系数 |
| **水平力模型** | 纯速度阻尼 | 深度×阻力系数 + 速度×耦合系数 + 姿态耦合项 |
| **参数化程度** | 部分硬编码 | 全部可配置参数 |
| **与论文的关系** | 名称相似但实现无关 | 基于论文思想的简化版本 |

---

## 六、参数含义详解

### 基础物理参数

| 参数 | 物理含义 | 说明 |
|-----|---------|-----|
| `sand_zeta` | 整体阻力系数 | 类似介质密度，越大阻力越大 |
| `foot_area` | 足端面积 | 与力成正比，面积越大阻力越大 |
| `foot_bottom_offset` | 脚底位置偏移 | 用于计算入沙深度的几何修正 |
| `min_sink_depth` | 最小入沙深度 | 只要有接触就有最小阻力 |
| `contact_force_threshold` | 接触力阈值 | 超过此值认为足端在沙中 |

### 增益参数（调参核心）

| 参数 | 控制什么 | 增大的效果 |
|-----|---------|-----------|
| `vertical_depth_gain` | 深度对垂直力的影响 | 同样深度下，垂直支撑力更大 |
| `vertical_speed_gain` | 向下速度对垂直力的影响 | 快速插入沙中时，缓冲/阻力更大 |
| `horizontal_depth_gain` | 深度对水平力的影响 | 埋得越深，水平移动越困难 |
| `horizontal_speed_gain` | 水平速度对水平力的影响 | 快速水平移动时，阻尼更大 |
| `posture_coupling_gain` | 姿态耦合项的强度 | 足端角度对水平力方向的影响更明显 |
| `upward_relief_gain` | 向上运动时的卸载程度 | 向上拔腿时，阻力减小的程度 |
| `force_limit` | 最大外力限制 | 防止力过大导致数值不稳定 |

### RFT 系数矩阵

```python
self.rft_generic_M = [0.206, 0.169, 0.212, 0.358, 0.055, -0.124, 0.253, 0.007, 0.088]
```

| 索引 | 用途 | 数学含义 |
|-----|------|---------|
| m[0] | 垂直力基础系数 | 常数项 |
| m[1] | 垂直力 cos(attack_angle) 系数 | 与法向余弦成正比 |
| m[2] | 垂直力 sin(attack_angle) 系数 | 与法向正弦成正比 |
| m[3] | 阻力基础系数 | 常数项 |
| m[4] | 阻力 cos(intrusion_angle) 系数 | 与侵入角余弦成正比 |
| m[5] | 阻力 sin(intrusion_angle) 系数 | 与侵入角正弦成正比 |
| m[6] | 耦合系数基础值 | 常数项 |
| m[7] | 耦合系数 cos(phase_angle) 系数 | 与相位角余弦成正比 |
| m[8] | 耦合系数 sin(phase_angle) 系数 | 与相位角正弦成正比 |

这些系数很可能来源于论文中的实验数据拟合！

---

## 七、理论基础

### Zero1 理论模型
```
Fz = ζ * A * (k1 * d + k2 * v_down - k3 * v_up)
Fxy = -ζ * k4 * vxy
```
- 简单的线性模型
- 类似阻尼器+弹簧系统
- 无角度依赖

### Zero3 理论模型
```
α = attack_angle
β = intrusion_angle
γ = β - α (phase_angle)

c_vertical = m[0] + m[1]*cos(α) + m[2]*sin(α)
c_drag = m[3] + m[4]*cos(β) + m[5]*sin(β)
c_coupling = m[6] + m[7]*cos(γ) + m[8]*sin(γ)

Fz = ζ * A * (k_vd * d * c_vertical + k_vs * v_down * (0.5 + c_coupling) - k_ur * v_up * c_drag)
Fxy = -ζ * A * v_dir * (k_hd * d * c_drag + k_hs * v_tan * (0.5 + c_coupling)) + ζ * A * k_pc * d * c_coupling * n_xy
```
- 基于角度的系数计算
- 类似论文思想的简化形式
- 引入姿态耦合项
- 水平力与深度相关

### 论文理论模型
论文 "A Terradynamics of Legged Locomotion on Granular Media" 的核心思想：
1. 将足端分解为小面元
2. 对每个面元，根据其法向、运动方向、深度查表求应力
3. 积分所有面元得到总力

Zero3 是：
- ✅ 保留了角度依赖的思想
- ✅ 保留了经验系数矩阵的思想
- ❌ 简化为单一点受力，无面元分解
- ❌ 简化为解析形式，无查表

---

## 八、总结

| 方面 | Zero2（平地） | Zero1（原始沙地） | Zero3（改进沙地） |
|-----|-------------|-----------------|-----------------|
| 阻力来源 | 仅物理引擎 | 物理引擎 + 代码阻尼 | 物理引擎 + 简化RFT |
| 理论基础 | 无 | 简单阻尼 | 论文思想简化 |
| 可调参数 | 无 | 4个 | 9个 + 9个RFT系数 |
| 与论文关系 | 无 | 名称相似 | 有实质性关系 |
| 实验位移（6秒） | ~0.3-0.4m | ~0.329m | ~0.169m |

---

## 九、关键改进点（Zero1 → Zero3）

1. ✅ **真正使用 `rft_generic_M` 矩阵**
2. ✅ **引入姿态角度计算**（attack_angle / intrusion_angle / phase_angle）
3. ✅ **水平力与深度相关**
4. ✅ **全部参数化**（移除硬编码数值）
5. ✅ **引入姿态耦合项**（posture_coupling）
6. ✅ **使用新的接口**（`permanent_wrench_composer`）
7. ✅ **添加外力限制**（`force_limit`）

