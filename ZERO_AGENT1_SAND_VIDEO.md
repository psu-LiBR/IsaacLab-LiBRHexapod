# Zero Agent1 Sand Video Recording Update

## 目标

将 `zero_agent1` 改造成：
- 直接使用六足沙地演示环境
- 调用 `HexapodSandPlayEnv`
- 录制真实机器人在沙地环境中的 RGB 视频并输出为 MP4

## 修改文件

1. `scripts/environments/zero_agent1.py`
2. `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/hexapod/hexapod_sand_play_env.py`

## 主要改动

### 1. `zero_agent1.py`

- 将默认任务改为 `Isaac-Velocity-Sand-Hexapod-Play-v0`
- 不再使用伪造沙丘和像素叠加的方式
- 改为直接实例化 `HexapodSandPlayEnv`
- 录制环境真实渲染出来的 RGB 帧，并写入 MP4

### 2. `hexapod_sand_play_env.py`

- 将足端索引从单个脚改为实际的六足足端 body 集合
- 让沙地接触、沉陷和外力计算针对所有足端生效
- 使得沙地环境中的机器人脚部可以正确参与地形交互

## 运行方式

```bash
./isaaclab.sh -p scripts/environments/zero_agent1.py --task Isaac-Velocity-Sand-Hexapod-Play-v0 --num_envs 1
```

输出视频路径：

```bash
my_videos/hexapod_sand_demo.mp4
```

## 验证结果

已使用下面的命令进行语法检查：

```bash
python3 -m py_compile scripts/environments/zero_agent1.py source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/hexapod/hexapod_sand_play_env.py
```

结果：通过，未出现语法错误。

## 最新状态（2026-07-13）

- 已修复 `HexapodSandPlayEnv` 在初始化阶段找不到足端 body 的问题，改为先尝试常见 `*_foot` / `*foot` 匹配，再回退到实际机器人 `body_names` 中的六足足端名称列表。
- 这一版把脚本和实际 USD 资产体名对齐，避免因为正则命中为空而直接中断环境初始化。
- 现在开始进行新的实际运行验证；若脚本继续报错，会把新的异常信息、视频路径和生成文件一并追加到这里。
