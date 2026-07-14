
# 六足机器人尺寸参数

## 测量方法
通过解析 `hexapod-assets/OBJ/` 目录下的OBJ文件获取各部件的顶点坐标，计算出每个部件的尺寸信息。

---

## 各部件详细尺寸

### 主体部件
| 部件名称 | x方向长度 (m) | y方向宽度 (m) | z方向高度 (m) |
|---------|-------------|-------------|-------------|
| CenterLink | 0.1340 | 0.0443 | 0.1057 |
| FrontLink | 0.1057 | 0.0875 | 0.0443 |
| BackLink | 0.1060 | 0.0965 | 0.0443 |

### 足端部件
所有六条腿的足端部件尺寸完全相同：
| 部件名称 | x方向长度 (m) | y方向宽度 (m) | z方向高度 (m) |
|---------|-------------|-------------|-------------|
| FrontLeft / FrontRight / MiddleLeft / MiddleRight / BackLeft / BackRight | 0.0290 | 0.0842 | 0.0423 |

---

## 整体尺寸估计

### 主体尺寸
| 尺寸参数 | 数值 (米) | 说明 |
|---------|---------|------|
| 体长（主体） | 0.3457 | FrontLink + CenterLink + BackLink 总长度 |
| 主体最大宽度 | 0.0965 | BackLink的y方向宽度 |
| 主体最大高度 | 0.1057 | CenterLink的z方向高度 |

### 整体尺寸估计（含腿）
| 尺寸参数 | 数值 (米) | 说明 |
|---------|---------|------|
| 总长度（体长） | **0.35** | 约35厘米，主体长度的近似值 |
| 总宽度（含展开的腿） | ~0.3 | 约30厘米，包括腿完全展开时的宽度 |
| 总高度（含站立的腿） | ~0.15 | 约15厘米，包括腿部站立时的高度 |

---

## 坐标系说明
- **x轴**：机器人前进方向（长度方向）
- **y轴**：机器人侧面方向（宽度方向）
- **z轴**：机器人高度方向

---

## 相关文件
- [hexapod-assets/OBJ/](file:///home/ubuntu/IsaacLab-LiBRHexapod/hexapod-assets/OBJ/) - 机器人各部件的OBJ文件
- [analyze_hexapod_dims.py](file:///home/ubuntu/IsaacLab-LiBRHexapod/analyze_hexapod_dims.py) - 尺寸分析脚本
- [MOVEMENT_TRACKING.md](file:///home/ubuntu/IsaacLab-LiBRHexapod/MOVEMENT_TRACKING.md) - 移动距离追踪功能说明
