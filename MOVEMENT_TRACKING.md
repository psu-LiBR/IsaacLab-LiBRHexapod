# 移动距离追踪功能

## 修改概述
为zero_agent1.py和zero_agent2.py添加了移动距离追踪功能，在6秒（300步）模拟结束后输出详细的移动统计信息。

## 修改的文件

1. **[`scripts/environments/zero_agent1.py`](file:///home/ubuntu/IsaacLab-LiBRHexapod/scripts/environments/zero_agent1.py)** - 沙地环境
2. **[`scripts/environments/zero_agent2.py`](file:///home/ubuntu/IsaacLab-LiBRHexapod/scripts/environments/zero_agent2.py)** - 平地环境

## 新增功能

### 追踪的信息
- 初始位置（x, y坐标）
- 最终位置（x, y坐标）
- x方向移动距离
- 二维平面总移动距离
- 移动了多少个体长（假设体长≈0.5米）

### 实现细节
```python
# 记录初始位置
initial_pos = None
final_pos = None
HEXAPOD_BODY_LENGTH = 0.5  # 估计六足机器人的体长（米）

# 在第0步获取初始位置
if sim_steps == 0:
    initial_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].clone()

# 在第299步获取最终位置
if sim_steps == 299:
    final_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].clone()

# 计算并输出统计信息
distance_x = abs(final_pos[0] - initial_pos[0])
distance_2d = torch.sqrt((final_pos[0] - initial_pos[0])**2 + 
                         (final_pos[1] - initial_pos[1])**2)
body_lengths = distance_2d / HEXAPOD_BODY_LENGTH
```

## 使用方法

运行沙地环境：
```bash
./isaaclab.sh -p scripts/environments/zero_agent1.py --task Isaac-Velocity-Sand-Hexapod-Play-v0 --num_envs 1 --headless --enable_cameras
```

运行平地环境：
```bash
./isaaclab.sh -p scripts/environments/zero_agent2.py --task Isaac-Velocity-Flat-Hexapod-Play-v0 --num_envs 1 --headless --enable_cameras
```

## 预期输出示例

```
============================================================
沙地环境（zero_agent1）移动统计
============================================================
总模拟时间：300步 = 6.0秒
初始位置：x=0.000m, y=0.000m
最终位置：x=1.234m, y=0.056m
x方向移动距离：1.234米
二维平面总移动距离：1.235米
相当于移动了：2.5个体长（体长≈0.5米）
============================================================
```

## 六足机器人尺寸说明
- 估计体长：0.5米（基于机器人结构估计）
- 如果你有更准确的尺寸数据，可以修改`HEXAPOD_BODY_LENGTH`变量的值
