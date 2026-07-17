
# Zero_Agent 系列步态分析

---

## 一、步态概述

zero_agent1/2/3 使用的是**基于正弦函数的波浪式前进步态**，通过：
1. 脊柱关节的反向摆动
2. 左右腿的反相交替摆动
3. 同侧腿的微小相位差

实现机器人向前移动。

---

## 二、机器人结构

### 2.1 关节编号
```
动作空间 (8维):
  [0] BackLink  (后脊柱)
  [1] FrontLink (前脊柱)
  [2] MiddleLeft  (左中腿)
  [3] MiddleRight (右中腿)
  [4] BackLeft    (左后腿)
  [5] BackRight   (右后腿)
  [6] FrontLeft   (左前腿)
  [7] FrontRight  (右前腿)
```

### 2.2 腿部布局
```
          FrontLeft [6]        FrontRight [7]
              ╲                      ╱
               ╲                    ╱
                ╲                  ╱
                 ╲                ╱
          MiddleLeft [2]      MiddleRight [3]
                   ╲            ╱
                    ╲          ╱
                     ╲        ╱
              BackLeft [4]  BackRight [5]

          ← 机器人前方
```

---

## 三、步态参数

| 参数 | 值 | 物理含义 |
|-----|----|---------|
| `gait_period` | 0.65 秒 | 步态周期时长 |
| `LEG_OFFSET` | -0.47 rad | 腿部关节默认角度（约 -27°） |
| `LEG_AMP` | 0.50 rad | 腿部摆动幅度（约 28.7°） |
| `SPINE_AMP` | 0.20 rad | 脊柱摆动幅度（约 11.5°） |
| `dt` | 0.02 秒 | 物理步长 |

---

## 四、步态逻辑详解

### 4.1 时间变量
```python
t = sim_steps * 0.02
omega = 2π * (t % gait_period) / gait_period
```
- `omega` 从 0 到 2π 循环变化
- 周期 = 0.65 秒 = 32.5 个物理步

---

### 4.2 脊柱关节（推动前进的关键）

| 关节 | 表达式 | 相位 | 作用 |
|-----|-------|------|------|
| BackLink [0] | `0.20 * sin(omega)` | 相位 0 | 后脊柱摆动 |
| FrontLink [1] | `0.20 * sin(omega + π)` | 相位 π（反相） | 前脊柱反向摆动 |

**说明**：
- 两个脊柱关节**反向摆动**
- 类似蛇的爬行，通过身体扭动产生推力

---

### 4.3 左侧腿（相位 A）

| 关节 | 表达式 | 相位偏移 |
|-----|-------|---------|
| MiddleLeft [2] | `LEG_OFFSET + 0.50*sin(omega)` | 0 |
| BackLeft [4] | `LEG_OFFSET + 0.50*sin(omega + 0.2π)` | +0.2π (超前) |
| FrontLeft [6] | `LEG_OFFSET + 0.50*sin(omega - 0.2π)` | -0.2π (滞后) |

**特点**：
- 左侧三条腿**大体同相**
- 但有 0.2π（约 36°）的微小相位差
- 形成“从前到后”或“从后到前”的波动

---

### 4.4 右侧腿（相位 B）

| 关节 | 表达式 | 相位偏移（相对于左侧） |
|-----|-------|----------------------|
| MiddleRight [3] | `LEG_OFFSET + 0.50*sin(omega + π)` | +π (反相) |
| BackRight [5] | `LEG_OFFSET + 0.50*sin(omega + 1.2π)` | +1.2π (反相 + 超前) |
| FrontRight [7] | `LEG_OFFSET + 0.50*sin(omega + 0.8π)` | +0.8π (反相 + 滞后) |

**特点**：
- 右侧腿与左侧腿**完全反相**（差 π）
- 右侧内部同样有 0.2π 相位差

---

## 五、相位关系图示

### 5.1 时间轴（0 到 2π）

```
omega (0 ~ 2π):
|--------------------|--------------------|
0                    π                    2π

左腿相位 (A):
MiddleLeft [2] : ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BackLeft   [4] :   ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FrontLeft  [6] :     ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

右腿相位 (B，反相):
MiddleRight[3] :         ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BackRight  [5] :           ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FrontRight [7] :             ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 5.2 相对相位表

| 关节 | 绝对相位 | 相对于 MiddleLeft 的相位差 |
|-----|---------|------------------------|
| BackLink (脊柱) | 0 | 0 |
| FrontLink (脊柱) | π | π |
| MiddleLeft | 0 | 0 |
| BackLeft | +0.2π | +0.2π |
| FrontLeft | -0.2π | -0.2π |
| MiddleRight | π | π |
| BackRight | π + 0.2π | π + 0.2π |
| FrontRight | π - 0.2π | π - 0.2π |

---

## 六、步态模式总结

### 6.1 整体模式
```
这不是标准的三角步态，而是：

┌─────────────────────────────────────┐
│  波浪式交替步态                     │
│  - 左右腿反相交替                   │
│  - 同侧腿有小相位差形成波动         │
│  - 脊柱反向摆动增加推进力           │
└─────────────────────────────────────┘
```

### 6.2 运动逻辑
1. **左右交替**：左腿抬高时右腿着地，反之亦然
2. **波浪式推进**：同侧腿的微小相位差让力从前向后（或从后向前）传递
3. **脊柱扭动**：脊柱反向摆动产生类似蛇行的推力

---

## 七、步态图解（时间切片）

### t = 0 时刻（omega = 0）
```
sin(0) = 0
sin(π) = 0
sin(±0.2π) = ±sin(36°) ≈ ±0.59

脊柱: BackLink=0, FrontLink=0（中立位）

左腿: MiddleLeft=LEG_OFFSET
      BackLeft=LEG_OFFSET + 0.50*0.59
      FrontLeft=LEG_OFFSET - 0.50*0.59

右腿: MiddleRight=LEG_OFFSET
      BackRight=LEG_OFFSET - 0.50*0.59
      FrontRight=LEG_OFFSET + 0.50*0.59
```

### t = π/2 时刻（omega = π/2）
```
sin(π/2) = 1
sin(3π/2) = -1

脊柱: BackLink=0.20, FrontLink=-0.20（反向扭动）

左腿: MiddleLeft=LEG_OFFSET + 0.50（前摆）
      BackLeft=LEG_OFFSET + 0.50*sin(0.7π)
      FrontLeft=LEG_OFFSET + 0.50*sin(-0.3π)

右腿: MiddleRight=LEG_OFFSET - 0.50（后摆）
      BackRight=LEG_OFFSET + 0.50*sin(1.7π)
      FrontRight=LEG_OFFSET + 0.50*sin(1.3π)
```

---

## 八、与标准步态的对比

| 步态类型 | 特点 | zero_agent 步态 |
|---------|------|----------------|
| 三角步态 (Tripod) | 3条腿支撑，3条腿摆动 | ❌ 不是 |
| 波动步态 (Wave) | 依次运动，类似爬行动物 | ⚠️ 有波动思想 |
| 爬行步态 (Crawl) | 一条一条动 | ❌ 不是 |
| 六足同相 | 六条腿一起动 | ❌ 不是 |

**zero_agent 步态分类**：
→ 自定义的“**左右交替 + 同侧波动 + 脊柱扭动**”步态

---

## 九、代码摘要

步态核心逻辑（在 [zero_agent3.py](file:///home/ubuntu/IsaacLab-LiBRHexapod/scripts/environments/zero_agent3.py#L106-L118)）：
```python
# 脊柱关节（反向摆动，推动前进）
q_target[0] = SPINE_AMP * torch.sin(omega)          # BackLink
q_target[1] = SPINE_AMP * torch.sin(omega + pi_tensor) # FrontLink (反向)

# 左侧腿（MiddleLeft[2], BackLeft[4], FrontLeft[6]）- 相位 A
q_target[2] = LEG_OFFSET + LEG_AMP * torch.sin(omega)               # MiddleLeft
q_target[4] = LEG_OFFSET + LEG_AMP * torch.sin(omega + 0.2*pi_tensor) # BackLeft
q_target[6] = LEG_OFFSET + LEG_AMP * torch.sin(omega - 0.2*pi_tensor) # FrontLeft

# 右侧腿（MiddleRight[3], BackRight[5], FrontRight[7]）- 相位 B（反相）
q_target[3] = LEG_OFFSET + LEG_AMP * torch.sin(omega + pi_tensor)    # MiddleRight
q_target[5] = LEG_OFFSET + LEG_AMP * torch.sin(omega + 1.2*pi_tensor) # BackRight
q_target[7] = LEG_OFFSET + LEG_AMP * torch.sin(omega + 0.8*pi_tensor) # FrontRight
```

---

## 十、总结

| 特性 | 说明 |
|-----|------|
| **驱动方式** | 正弦函数驱动 |
| **周期** | 0.65 秒 |
| **脊柱** | 反向摆动，产生推力 |
| **左右关系** | 完全反相交替 |
| **同侧腿相位差** | ±0.2π，形成波动 |
| **腿部幅度** | 0.50 rad (约 28.7°) |
| **脊柱幅度** | 0.20 rad (约 11.5°) |

这个步态通过左右交替 + 同侧波动 + 脊柱扭动，实现了机器人的稳定前进！

