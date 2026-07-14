# RFT 力参数优化

## 问题描述
**问题1**：在沙地环境中，RFT（阻力理论）产生的力过大，导致机器人出现“弹跳飞起”的现象，与真实沙地物理不符。

**问题2**：第一轮优化后发现RFT力完全没有生效（max_depth=0.0000, max_fz=0.000），因为深度计算逻辑有问题。

**问题3**：第二轮修复深度计算后，`max_depth` 有值了，但 `max_fz` 还是 0.000，因为力的计算系数太小了，导致实际力值太小（只有约 0.00003N），在三位小数下显示为 0！

## 问题根源分析

### 原始参数问题
| 参数 | 原值 | 问题 |
|------|------|------|
| `sand_zeta` | 2.0 | 阻力系数过大 |
| `foot_area` | 0.01 | 足端面积过大 |
| `foot_bottom_offset` | 0.03 | 第一次优化改成 0.015，但仍然不够大 |
| `min_sink_depth` | 0.002 | 第一次优化设为 0.0，导致没有接触力 |
| 垂直力乘数 | 20.0 | 垂直方向力系数过大 |
| 水平阻力系数 | 30.0 | 水平阻尼过大 |
| 垂直力条件 | `downward_speed > 0` | 只在向下速度时才施加力 |
| 深度计算 | 简单高度计算 | 无法正确处理足端高于地面的情况 |

## 优化方案（三轮）

### 第一轮：降低力的大小
- 降低 sand_zeta、foot_area、垂直/水平力系数
- 改进垂直力计算逻辑

### 第二轮：修复深度计算，使RFT力能够生效
从调试日志发现：
- `min_foot_bottom_h` ≈ 0.04m（都是正数，脚在地面以上）
- `max_depth=0.0000`，完全没有入沙深度
- `max_fz=0.000`，RFT力完全没生效

**解决方案**：
1. 大幅增加 `foot_bottom_offset` 到 0.06m，确保即使脚在地面之上也能计算深度
2. 重新引入小的 `min_sink_depth` (0.001m)，确保有接触就有力
3. 改进深度计算逻辑，即使脚在地面之上也能产生合理的深度值

### 第三轮：调整力的大小，使其既有效果又不过大
从第二轮调试日志发现：
- `max_depth` 有值（0.001-0.006m）了
- 但是 `max_fz` 仍然是 0.000，因为力计算得太小了
- 计算：0.35 * 0.003 * (0.006 * 5.0) = 0.0000315N，太小了！

**解决方案**：
1. 适度增加 `sand_zeta` 到 0.5
2. 把 `foot_area` 改回 0.01（更合理的足端面积）
3. 大幅增加垂直力乘数（从 5.0 → 50.0）和速度项乘数
4. 适度增加水平阻力系数（从 5.0 → 20.0）

### 修改文件
1. `hexapod_sand_play_env.py`
2. `hexapod_sand_train_env.py`

### 最终优化参数

| 参数 | 初始值 | 第一轮 | 第二轮 | 最终值 | 说明 |
|------|--------|--------|--------|--------|------|
| `sand_zeta` | 2.0 | 0.35 | 0.35 | 0.5 | 适度增大阻力系数 |
| `foot_area` | 0.01 | 0.003 | 0.003 | 0.01 | 更合理的足端面积 |
| `foot_bottom_offset` | 0.03 | 0.015 | 0.06 | 0.06 | 增大偏移确保能计算深度 |
| `min_sink_depth` | 0.002 | 0.0 | 0.001 | 0.001 | 小的最小深度确保有接触就有力 |
| 垂直力乘数 | 20.0 | 5.0 | 5.0 | 50.0 | 增大到合理范围 |
| 垂直速度乘数 | 2.0 | 1.0 | 1.0 | 10.0 | 增大到合理范围 |
| 水平阻力系数 | 30.0 | 5.0 | 5.0 | 20.0 | 适度增大 |
| 向上速度乘数 | 无 | 0.5 | 0.5 | 2.0 | 适度增大 |
| 垂直力条件 | `downward_speed > 0` | 始终应用 | 始终应用 | 始终应用 | 在整个接触期间都有垂直力 |

### 垂直力计算优化（第三轮）
```python
# 向下运动时提供支撑/阻力，向上运动时也有轻微阻力
vertical_force = zeta * area * (depth * 50.0 + downward_speed * 10.0)
# 向上运动时也有轻微阻力
vertical_force -= zeta * area * upward_speed * 2.0
```

### 水平阻力优化（第三轮）
```python
# 适度增大阻尼系数
forces[..., 0:2] = -zeta * 20.0 * foot_vel[..., 0:2]
```

### 深度计算优化（第二轮修复）
```python
# 即使脚底高度略高于0，只要有接触，就给一个基础深度
foot_bottom_height = foot_positions[..., 2] - foot_bottom_offset
depth_from_height = torch.clamp(-foot_bottom_height, min=0.0)
# 对于有接触但深度为0的情况，给一个最小深度
base_depth = torch.where(
    in_sand_mask,
    min_sink_depth + torch.clamp(-foot_bottom_height * 0.5, min=0.0),
    torch.zeros_like(foot_bottom_height),
)
depth = torch.maximum(depth_from_height, base_depth)
```

## 预期效果
1. 机器人在沙地中行走更自然
2. 不会出现“弹跳飞起”的现象
3. 水平移动会有合理的阻力，但不会过度减速
4. 垂直方向有适当的支撑力，既不会过度下沉也不会弹起

## 验证命令
```bash
# 测试沙地环境
./isaaclab.sh -p scripts/environments/zero_agent1.py --task Isaac-Velocity-Sand-Hexapod-Play-v0 --num_envs 1 --headless --enable_cameras

# 对比平地环境
./isaaclab.sh -p scripts/environments/zero_agent2.py --task Isaac-Velocity-Flat-Hexapod-Play-v0 --num_envs 1 --headless --enable_cameras
```
