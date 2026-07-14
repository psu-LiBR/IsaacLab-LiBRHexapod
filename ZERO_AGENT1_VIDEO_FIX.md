# Zero Agent1 视频录制全黑问题修复

## 问题描述
运行 `zero_agent1.py` 时，录制的 MP4 视频全是黑色的，没有显示六足机器人在沙地环境中的画面。

## 问题根源
1. 代码直接调用 `env.unwrapped.sim.render_to_array()` 这个不存在的方法
2. 导致获取的帧数据始终为 `None`，然后代码生成全黑的 numpy 数组填充视频
3. 同时还试图手动实例化 `HexapodSandPlayEnv`，而不是通过 gym 注册表统一创建
4. **关键问题**：在 headless 模式下运行时，没有启用 `--enable_cameras` 参数！

## 修改文件
- `scripts/environments/zero_agent1.py`

## 主要修复点
1. **移除手动渲染代码**
   - 删除了对 `env.unwrapped.sim.render_to_array()` 的调用
   - 移除了 `imageio` 相关的手动视频写入逻辑
   - 删除了复杂的帧数据处理代码

2. **使用标准 Gymnasium 工具**
   - 采用 `gym.wrappers.RecordVideo` 包装器自动处理视频录制
   - 通过 `gym.make()` 统一创建环境
   - 指定 `render_mode="rgb_array"` 启用正确的渲染模式

3. **简化代码结构**
   - 移除了不必要的 `HexapodSandPlayEnv` 直接实例化
   - 移除了不再需要的 imports（numpy, math 等）

4. **优化相机视角**
   - 参考 play_sand_headless.py 设置更合适的相机视角

5. **关键修复：设置渲染模式和 RTX 传感器**（最重要！）
   - 在 AppLauncher 后立即使用 `carb.settings.get_settings().set_bool("/isaaclab/render/rtx_sensors", True)`（关键时序！必须在创建环境之前）
   - 使用 `--enable_cameras` 参数自动设置 `render_mode=PARTIAL_RENDERING`

## 修改后的运行方式
**关键！必须加上 --enable_cameras 参数！**

```bash
./isaaclab.sh -p scripts/environments/zero_agent1.py --task Isaac-Velocity-Sand-Hexapod-Play-v0 --num_envs 1 --headless --enable_cameras
```

## 验证
已使用 `python3 -m py_compile` 对修改后的文件进行语法检查，未发现语法错误。

## 参考
- 参考项目中正常工作的脚本：`scripts/environments/zero_agent.py`
- 参考项目中正常工作的脚本：`scripts/environments/play_sand_headless.py`

## 2026-07-14 追加修改：方案 A，定位真实足端 body

### 背景
- 在 `zero_agent1.py` 的沙地环境运行日志中，RFT 调试输出持续为：
  - `max_depth=0.0000`
  - `max_fz=0.000`
- 这说明 RFT 代码虽然被调用了，但当前没有产生任何有效外力。

### 进一步分析
- 原实现中，`HexapodSandPlayEnv` 和 `HexapodSandTrainEnv` 的足端索引使用硬编码：
  - `[2, 3, 4, 5, 6, 7]`
- 这只能说明“假设这些 body 索引是足端”，但无法确认它们是否真的是机器人实际接触地面的 6 个刚体。
- 因此采用 **方案 A**：
  - 不再依赖硬编码 body index
  - 改为通过 `robot.find_bodies([...])` 按名称解析真实足端 body

### 本次修改文件
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/hexapod/hexapod_sand_play_env.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/hexapod/hexapod_sand_train_env.py`

### 本次修改内容
1. 新增真实足端名称列表：
   - `["MiddleLeft", "MiddleRight", "BackLeft", "BackRight", "FrontLeft", "FrontRight"]`
2. 在 `_init_feet()` 中改为：
   - `robot = self.scene["robot"]`
   - `foot_indices, resolved_names = robot.find_bodies(self.foot_body_names)`
3. 增加解析失败保护：
   - 如果解析数量与预期不一致，直接抛出 `RuntimeError`
4. 增加调试输出：
   - 打印真实足端 body 名称
   - 打印真实足端 body 索引

### 目的
- 确认 RFT 当前施加外力的对象，是否真的是机器人真实的 6 个足端刚体。
- 这是修复 `depth` 始终为 0 的第一步。

### 当前结论
- 从目前日志可以确认：RFT 调用链已经跑通，但外力为 0。
- 本次修改的目标不是直接修复 RFT 数值，而是先确认“施力对象是否正确”。

## 2026-07-14 追加修改：方案 B，改为接触触发 + 脚底偏移近似的 depth 计算

### 背景
- 采用方案 A 后，已经确认真实足端 body 成功解析为：
  - `MiddleLeft`
  - `MiddleRight`
  - `BackLeft`
  - `BackRight`
  - `FrontLeft`
  - `FrontRight`
- 但运行日志仍然显示：
  - `max_depth=0.0000`
  - `max_fz=0.000`
- 说明问题已经不在“施力对象错误”，而在于 `depth` 的定义方式不合理。

### 问题根源
- 原实现使用：
  - `depth = clamp(foot_radius - body_pos_z, min=0)`
- 这里的 `body_pos_z` 是足端 body 原点的世界坐标高度，不等于脚底真正接触地面的高度。
- 即使足端已经接地，该 body 原点仍可能远高于地面，因此 `depth` 会长期为 0。

### 本次修改文件
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/hexapod/hexapod_sand_play_env.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/hexapod/hexapod_sand_train_env.py`

### 本次修改内容
1. 引入 `contact_forces` 传感器参与 RFT 触发判定
   - 使用 `contact_sensor.data.net_forces_w_history` 读取足端接触力历史
   - 当接触力大于阈值时，才认为足端真正“入沙”
2. 新增脚底偏移近似参数
   - `foot_bottom_offset = 0.12`
   - 用于把足端 body 原点高度转换为“脚底近似高度”
3. 新增最小入沙深度
   - `min_sink_depth = 0.01`
   - 只要检测到接触，就至少给一个非零 depth，确保 RFT 能真正产生力
4. 新的 depth 逻辑
   - `foot_bottom_height = body_pos_z - foot_bottom_offset`
   - `depth_from_height = clamp(-foot_bottom_height, min=0)`
   - 最终 `depth = max(depth_from_height, min_sink_depth)`，但仅在接触时启用
5. 扩展调试输出
   - 新增：
     - `max_contact`
     - `min_foot_bottom_h`
   - 现在可以同时判断：
     - 足端是否真的接触地面
     - 脚底近似高度是否低于地面
     - RFT 是否已经产生非零外力

### 目的
- 让 RFT 不再依赖不可靠的“body 原点 z < 常数”条件。
- 先稳定地产生非零外力，确认 RFT 阻力链路能够真实影响机器人运动。

### 预期现象
- 再次运行 `zero_agent1.py` 后，调试输出中应至少出现：
  - `max_contact > 0`
  - `max_depth > 0`
  - `max_fz > 0`
- 如果这三项仍然为 0，则说明还需要继续下探接触数据或脚底偏移参数。

## 2026-07-14 追加修改：抑制“弹飞”，改成更偏拖拽的沙地阻力

### 背景
- 方案 B 生效后，日志中已经首次出现：
  - `max_contact > 0`
  - `max_depth > 0`
  - `max_fz > 0`
- 说明 RFT 已经开始对机器人施加非零外力。
- 但视频表现为机器人被明显“顶起来/弹起来”，不像沙地，更像弹簧地板。

### 原因分析
- 上一版参数把脚底偏移估得过大：
  - `foot_bottom_offset = 0.12`
- 同时最小入沙深度也偏大：
  - `min_sink_depth = 0.01`
- 再加上垂向力始终存在，导致一接触地面就有较持续的向上支撑，从而出现“被弹起”的效果。

### 本次修改文件
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/hexapod/hexapod_sand_play_env.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/hexapod/hexapod_sand_train_env.py`

### 本次修改内容
1. 减小脚底偏移
   - `foot_bottom_offset: 0.12 -> 0.03`
2. 减小最小入沙深度
   - `min_sink_depth: 0.01 -> 0.002`
3. 修改垂向力策略
   - 不再“只要接触就持续给向上力”
   - 改为仅在足端**向下压地**时（`downward_speed > 0`）才施加垂向支撑
4. 削弱垂向支撑强度
   - 垂向力系数从更强的版本降为：
     - `depth * 20.0 + downward_speed * 2.0`
5. 增强水平拖拽
   - 水平阻力从 `-sand_zeta * 20.0 * foot_vel_xy`
   - 调整为 `-sand_zeta * 30.0 * foot_vel_xy`

### 目标
- 减少“被地面弹起来”的现象
- 让 Sand 环境更像：
  - 前进时被拖拽
  - 行走更吃力
  - 而不是被向上顶飞

### 新的预期现象
- `max_contact`、`max_depth`、`max_fz` 仍然应为非零
- 但机器人整体姿态应比上一版更稳定
- 视频中应更容易看到“拖拽感”，而不是明显离地弹跳
