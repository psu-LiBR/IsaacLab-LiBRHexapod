# 沙地环境注册问题修复

## 问题描述
1. 初始问题：运行 `Isaac-Velocity-Sand-Hexapod-Play-v0` 时，虽然视频能正常录制，但实际上使用的是标准的 `ManagerBasedRLEnv`，而不是自定义的 `HexapodSandPlayEnv`。
2. 后续问题：修改 `entry_point` 后，出现 `TypeError: HexapodSandPlayEnv.__init__() got an unexpected keyword argument 'env_cfg_entry_point'` 错误。

## 问题根源

### 第一阶段
在 `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/hexapod/__init__.py` 中，沙地环境的注册配置错误地将 `entry_point` 指向了标准环境类：

```python
# 错误的配置
gym.register(
    id="Isaac-Velocity-Sand-Hexapod-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",  # ❌ 错误！
    ...
)
```

### 第二阶段
修改 `entry_point` 指向自定义类后，Gym 注册时会将所有 `kwargs`（包括 `env_cfg_entry_point`、`rsl_rl_cfg_entry_point` 等）直接传递给构造函数，但自定义类的构造函数没有准备好接收这些额外参数。

另外，`hexapod_sand_train_env.py` 中的类名是 `HexapodsSandPlayEnv`（复数），与注册时使用的 `HexapodSandTrainEnv` 不匹配。

## 修复方案

### 1. 修改环境注册
**文件**：`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/hexapod/__init__.py`

将两个沙地环境的 `entry_point` 分别指向正确的自定义环境类：

1. **训练环境**：`Isaac-Velocity-Sand-Hexapod-v0`
   - 从：`"isaaclab.envs:ManagerBasedRLEnv"`
   - 改为：`"isaaclab_tasks.manager_based.locomotion.velocity.config.hexapod.hexapod_sand_train_env:HexapodSandTrainEnv"`

2. **演示环境**：`Isaac-Velocity-Sand-Hexapod-Play-v0`
   - 从：`"isaaclab.envs:ManagerBasedRLEnv"`
   - 改为：`"isaaclab_tasks.manager_based.locomotion.velocity.config.hexapod.hexapod_sand_play_env:HexapodSandPlayEnv"`

### 2. 修复自定义环境构造函数
**文件 1**：`hexapod_sand_play_env.py`
- 在 `__init__` 方法中添加 `**kwargs` 参数来接收额外参数

**文件 2**：`hexapod_sand_train_env.py`
- 修复类名从 `HexapodsSandPlayEnv` 改为 `HexapodSandTrainEnv`
- 在 `__init__` 方法中添加 `**kwargs` 参数

### 3. 修改视频文件名（附加）
**文件**：`scripts/environments/zero_agent1.py`
- 在 `RecordVideo` 包装器中添加 `name_prefix="hexapod_sand_demo"` 参数

## 自定义环境的功能
现在正确使用 `HexapodSandPlayEnv` 后，将获得以下增强功能：

1. **实时沙地塌陷**：当机器人足端踩入沙地时，周围的高度场会实时降低，产生视觉上的沙坑
2. **RFT 阻力计算**：基于 Resistive Force Theory 计算沙地对足端的阻力
3. **外力注入**：将计算出的阻力显式注入到 PhysX 物理引擎中

## 验证
现在再次运行以下命令时，将正确使用自定义的沙地环境：

```bash
./isaaclab.sh -p scripts/environments/zero_agent1.py --task Isaac-Velocity-Sand-Hexapod-Play-v0 --num_envs 1 --headless --enable_cameras
```

你应该能在日志中看到来自 `HexapodSandPlayEnv` 的特定输出，例如：
- `[INFO]: 🚀 地形场景解析链彻底打通！高度场沙丘已成功实例化并渲染入视窗！`
- `[INFO]: 已定位六足机器人足端 body: [...]`

生成的视频文件名将是 `hexapod_sand_demo-step-0.mp4`。
