# 六足机器人三角步态对比实验（zero_agent2）

## 日期
2026-07-14

## 改动概述

本次改动创建了 `zero_agent2.py` 并更新了 `zero_agent1.py`，目的是使用相同的标准三角步态，对比六足机器人在平地和沙地环境中的行走效果。

## 修复和优化记录

### 2026-07-14: 修复类型错误
- **问题**: `TypeError: sin(): argument 'input' (position 1) must be Tensor, not float`
- **原因**: `omega` 是 Python float 类型，`torch.sin()` 需要 Tensor 类型
- **修复**:
  1. 将 `omega` 转换为 Tensor：`omega = torch.tensor(..., device=env.unwrapped.device)`
  2. 将 `torch.pi` 也转换为 Tensor：`torch.tensor(torch.pi, device=env.unwrapped.device)`
- **影响文件**:
  - `scripts/environments/zero_agent1.py`
  - `scripts/environments/zero_agent2.py`

### 2026-07-14: 优化三角步态让机器人真正往前走
- **问题**: 机器人只是两边晃，不会往前走
- **优化内容**:
  1. **增大摆动幅度**: LEG_AMP 从 0.30 增加到 0.50，SPINE_AMP 从 0.10 增加到 0.15
  2. **调整步态周期**: gait_period 从 0.5 秒调整到 0.6 秒，稍微减慢一点让步态更稳定
  3. **优化相位关系**: 同一三脚架内的不同腿有轻微的相位差（±0.1π），产生更协调的前进步态
  4. **使用预定义变量**: 添加 `pi_tensor` 避免重复创建 Tensor
- **优化原理**:
  - 通过增大摆动幅度让机器人腿部动作更明显
  - 通过同一三脚架内腿的相位差，产生类似"波浪"的协调运动，推动机器人前进
  - 脊柱反向摆动和腿部摆动配合，增强前进动力
- **影响文件**:
  - `scripts/environments/zero_agent1.py`
  - `scripts/environments/zero_agent2.py`

## 文件清单

### 1. 新创建文件
- `scripts/environments/zero_agent2.py`: 平地环境测试脚本

### 2. 更新文件
- `scripts/environments/zero_agent1.py`: 沙地环境测试脚本（更新步态）

## 详细改动

### zero_agent2.py 主要特性：
- **环境**: `Isaac-Velocity-Flat-Hexapod-Play-v0`（原始平地环境）
- **步态**: 标准三角步态（Tripod Gait）
- **视频文件名前缀**: `hexapod_flat_demo`
- **步态参数**:
  - 步态周期: 0.5 秒
  - 腿部摆动幅度: 0.30 rad
  - 脊柱摆动幅度: 0.10 rad
  - 腿部默认位置: -0.47 rad

### zero_agent1.py 更新内容：
- **步态更新**: 从简单的原地周期性步态改为标准三角步态
- **保持沙地环境**: `Isaac-Velocity-Sand-Hexapod-Play-v0`
- **保持视频文件名前缀**: `hexapod_sand_demo`
- **保持沙地环境特定功能**: 沙地塌陷和 RFT 阻力

## 标准三角步态实现

### 关节顺序
```
0: BackLink (spine)
1: FrontLink (spine)
2: MiddleLeft
3: MiddleRight
4: BackLeft
5: BackRight
6: FrontLeft
7: FrontRight
```

### 三脚架分组
- **三脚架 A**: MiddleRight[3], BackLeft[4], FrontLeft[6]
- **三脚架 B**: MiddleLeft[2], BackRight[5], FrontRight[7]

### 步态原理
- 两个三脚架交替摆动
- 脊柱反向摆动增强前进动力
- 基于项目中 `hexapod_mimic_motion.py` 的标准实现

### 动作空间转换
由于环境使用 `JointPositionActionCfg` 配置：
- `scale=0.5`
- `use_default_offset=True`

动作计算方式：
```python
action = (q_target - q_default) / scale
```

## 运行命令

### 沙地环境（zero_agent1）
```bash
./isaaclab.sh -p scripts/environments/zero_agent1.py --task Isaac-Velocity-Sand-Hexapod-Play-v0 --num_envs 1 --headless --enable_cameras
```

### 平地环境（zero_agent2）
```bash
./isaaclab.sh -p scripts/environments/zero_agent2.py --task Isaac-Velocity-Flat-Hexapod-Play-v0 --num_envs 1 --headless --enable_cameras
```

## 输出文件

- **沙地视频**: `my_videos/hexapod_sand_demo-step-0.mp4`
- **平地视频**: `my_videos/hexapod_flat_demo-step-0.mp4`

## 实验目的
对比在相同步态下：
1. 沙地环境的塌陷效果
2. 平地环境的正常行走效果
3. 两者的差异

## 参考资料
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/hexapod/hexapod_mimic_motion.py`: 标准三角步态实现
