
# RFT 参数合理化调整指南

## 概述
本文档提供一个系统化的 RFT 参数调整方案，让参数更有物理意义，便于理解和校准。

---

## 1. 当前参数的问题

### 1.1 当前参数结构

```python
# 当前实现（问题点）
self.vertical_depth_gain = 10000.0    # 黑箱参数，物理意义不明确
self.horizontal_depth_gain = 8000.0   # 黑箱参数
self.sand_zeta = 5.0                  # 与 gain 混合，单位混乱
```

**问题**：
1. ❌ `gain` 参数物理意义不明确
2. ❌ `sand_zeta` 与 `gain` 相乘，单位/量级混乱
3. ❌ 难以根据物理直觉调整

---

## 2. 更合理的参数化方案

### 2.1 方案一：基于物理量纲的参数化（推荐）

#### 2.1.1 力的物理量纲

```
真实物理力应该具有量纲：
  F = ζ * A * d * g  (与浮力类似)
  其中：
    ζ：无量纲阻力系数 (0-1)
    A：面积 (m²)
    d：深度 (m)
    g：重力加速度 (9.81 m/s²)
```

#### 2.1.2 重新设计的参数结构

```python
# 建议的参数结构
self.sand_properties = {
    # 物理属性（来自论文或实验）
    'zeta_vertical': 0.4,       # 垂直阻力系数 (无量纲)
    'zeta_horizontal': 0.6,     # 水平阻力系数 (无量纲)
    'density': 1600.0,          # 沙的密度 (kg/m³)
    'gravity': 9.81,            # 重力加速度 (m/s²)

    # 动态特性
    'viscous_damping': 100.0,   # 粘性阻尼系数 (N·s/m)
    'impact_stiffness': 5000.0, # 冲击刚度 (N/m)
    'release_gain': 0.1,        # 释放时的卸载系数

    # 姿态耦合
    'posture_coupling': 0.05    # 姿态耦合强度 (0-1)
}
```

#### 2.1.3 力计算公式调整

```python
# 新的力计算（更物理）
vertical_force = (
    # 静支撑力（浮力类）
    sand_props['density'] * sand_props['gravity'] * self.foot_area * depth
    * sand_props['zeta_vertical'] * vertical_coeff

    # 动态阻尼力
    + sand_props['viscous_damping'] * downward_speed * (0.5 + coupling_coeff)

    # 向上卸载
    - sand_props['release_gain'] * upward_speed * drag_coeff
)

horizontal_force = (
    # 水平阻力
    -sand_props['density'] * sand_props['gravity'] * self.foot_area * depth
    * sand_props['zeta_horizontal'] * drag_coeff * tangential_dir

    # 水平粘性阻尼
    -sand_props['viscous_damping'] * tangential_speed * tangential_dir

    # 姿态耦合
    + sand_props['posture_coupling'] * ...
)
```

**优点**：
- ✅ 每个参数都有明确的物理意义
- ✅ 可以从文献/实验中获取合理范围
- ✅ 便于参数敏感性分析

---

## 3. 实际参数调整策略（分步进行）

### 3.1 第一步：先让深度计算更准确

**问题**：从之前的日志看，`max_depth` 只有 0.001m，太小了！

**解决方案**：
- 检查足端初始位置是否在地面上方太多
- 调整 `foot_bottom_offset` 或修改深度计算逻辑
- 目标：`max_depth` 应该在 0.01m - 0.05m 范围

---

### 3.2 第二步：固定基础参数，只调一个阻力系数

**建议操作**：
1. 暂时禁用姿态耦合（`posture_coupling_gain = 0`）
2. 暂时禁用速度项（只保留深度项）
3. 只调整一个参数，例如 `sand_zeta`，观察效果
4. 目标：获得明显但不过度的阻力

---

### 3.3 第三步：逐步引入其他效果

| 阶段 | 参数 | 目标 |
|------|-----|------|
| 1 | 垂直深度项 | 有明显的支撑感 |
| 2 | 水平深度项 | 有明显的前进阻力 |
| 3 | 垂直速度项 | 插入沙中有缓冲感 |
| 4 | 水平速度项 | 快速移动时有额外阻力 |
| 5 | 姿态耦合 | 轻微的方向依赖 |

---

## 4. 建议的参数范围（基于物理直觉）

### 4.1 合理参数范围

| 参数 | 建议范围 | 说明 |
|-----|---------|------|
| sand_zeta | 0.5 - 2.0 | 阻力强度，参考论文中典型值 |
| vertical_depth_gain | 1000 - 5000 | 支撑刚度，过大弹飞，过小下陷 |
| horizontal_depth_gain | 500 - 3000 | 水平阻力，参考垂直的 0.3-0.6 倍 |
| vertical_speed_gain | 50 - 200 | 阻尼，快速插入时的缓冲 |
| horizontal_speed_gain | 30 - 100 | 水平阻尼 |
| posture_coupling_gain | 0.0 - 0.5 | 轻微耦合，避免额外推进 |
| force_limit | 10 - 100 | 保护仿真稳定性 |

### 4.2 当前参数评估

| 参数 | 当前值 | 评估 | 建议 |
|-----|------|------|------|
| sand_zeta | 5.0 | ⚠️ 可能偏大 | 可尝试降低到 1.0-3.0 |
| vertical_depth_gain | 10000.0 | ⚠️ 可能偏大 | 可尝试 3000-6000 |
| horizontal_depth_gain | 8000.0 | ⚠️ 可能偏大 | 可尝试 2000-4000 |
| posture_coupling_gain | 1.0 | ✅ 合理 | 保持或略微增加 |

---

## 5. 调整建议（下一步）

### 5.1 短期建议（保持当前结构，微调）

```python
# 建议的微调（保持当前代码结构）
self.sand_zeta = 2.0  # 降低，避免过大
self.vertical_depth_gain = 5000.0  # 降低
self.horizontal_depth_gain = 3000.0  # 降低
self.posture_coupling_gain = 0.2  # 略微增加，引入一点耦合
```

### 5.2 中期建议（代码结构优化）

重构力计算，让参数更物理：
1. 分开"静"和"动"的部分
2. 给每个参数加注释说明物理意义
3. 建立参数配置字典，便于管理

---

## 6. 验证指标

每次调整参数后，检查以下指标：

| 指标 | 理想范围 |
|-----|---------|
| 6秒位移（沙地） | 0.05 - 0.15m |
| max_fz | 1 - 5N |
| max_depth | 0.01 - 0.05m |
| 视觉效果 | 机器人有下陷感，但不卡死 |

---

## 7. 总结

当前参数虽然效果不错（位移下降76.5%），但结构上可以更优化：
1. 降低过大的 gain 参数，避免数值不稳定
2. 考虑更物理的参数化方式（基于密度、gravity等）
3. 分步调整，每次只调一个参数
4. 验证时关注多个指标，不只是位移

---

*文档创建日期：2026-07-15*

