# 🎸 iGuitar - 吉他学习辅助系统

> 基于计算机视觉的吉他教学辅助工具，使用 MediaPipe 实时检测手部动作，帮助初学者纠正指法

![Version](https://img.shields.io/badge/version-0.5-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![MediaPipe](https://img.shields.io/badge/mediapipe-latest-orange)

---

## 📖 项目简介

iGuitar 是一个面向吉他初学者的智能学习辅助系统。通过摄像头实时捕捉用户的手部动作，与标准指法进行对比，并给出即时反馈，帮助学习者快速掌握正确的弹奏姿势。

### 核心功能（v0.5）

- ✅ 实时手部关键点检测（21个关键点）
- ✅ 双手同时追踪
- ✅ 骨架可视化显示
- ✅ 指板坐标映射与标定
- ✅ 和弦识别与反馈
- ✅ 歌曲练习模式
- ✅ FPS 性能监控
- ✅ **Web UI 界面**（基于 Gradio）
- ✅ 练习统计与数据分析
- ✅ 多首歌曲支持



---

## 🚀 快速开始

### 环境要求

- Python 3.8 或更高版本
- 摄像头设备
- 推荐 4GB+ 内存

### 安装步骤

1. **克隆项目**

```bash
git clone https://github.com/yourusername/iGuitar.git
cd iGuitar
```

2. **创建虚拟环境**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

3. **安装依赖**

```bash
pip install -r requirements.txt
```

或手动安装：

```bash
pip install opencv-python
pip install mediapipe
pip install numpy
pip install gradio
```

4. **下载模型文件**

创建 `assets/models` 目录并下载 MediaPipe 手部检测模型：

```bash
mkdir -p assets/models
cd assets/models

# Windows PowerShell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "hand_landmarker.task"

# Mac/Linux
curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

cd ../..
```

或手动下载：[下载链接](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)

将文件放到 `iGuitar/assets/models/hand_landmarker.task`

5. **运行标定**

首次使用需要进行指板标定：

```bash
python scripts/calibrate.py
```

按照屏幕提示，用食指依次触碰四个标定点。

6. **运行主程序**

**方式一：Web UI（推荐）**

```bash
python scripts/webui.py
```

启动后在浏览器打开显示的地址（通常是 `http://localhost:7860`）

**方式二：传统窗口模式**

```bash
python scripts/main.py
```

---

## 📂 项目结构

```
iGuitar/
├── src/                         # 源代码
│   ├── core/                   # 核心功能模块
│   │   ├── camera.py           # 摄像头输入
│   │   ├── hand_tracker.py     # 手部追踪
│   │   └── preprocessor.py     # 视频预处理
│   ├── mapping/                # 坐标映射
│   │   ├── calibrator.py       # 标定器
│   │   └── fretboard_mapper.py # 指板映射
│   ├── data/                   # 数据模块
│   │   ├── chord_db.py         # 和弦数据库
│   │   └── song.py             # 歌曲管理
│   └── ui/                     # UI显示
│       └── display.py          # 显示模块
├── scripts/                    # 可执行脚本
│   ├── main.py                # 主程序（窗口模式）
│   ├── webui.py               # Web UI 界面
│   └── calibrate.py           # 标定脚本
├── assets/                     # 资源文件
│   ├── models/                # ML模型
│   │   └── hand_landmarker.task
│   └── songs/                 # 歌曲文件
│       ├── twinkle.json
│       ├── happy_birthday.json
│       ├── ordinary_road.json
│       └── simple_practice.json
├── requirements.txt            # 依赖列表
└── README.md                   # 项目文档
```

---

## 🎮 使用说明

### 标定流程

1. 运行 `python scripts/calibrate.py`
2. 用食指依次触碰四个标定点：
   - 第1弦第1品
   - 第6弦第1品
   - 第1弦第12品
   - 第6弦第12品
3. 每个点按空格键记录
4. 完成后生成 `calibration_matrix.npy`

### Web UI 使用（推荐）

1. 运行 `python scripts/webui.py`
2. 在浏览器打开显示的地址
3. 选择歌曲并点击"开始练习"
4. 实时查看视频反馈和练习统计
5. 使用界面按钮控制：
   - **下一个** - 下一个和弦/音符
   - **上一个** - 上一个和弦/音符
   - **重置** - 回到开头
   - **停止** - 停止练习

### 传统窗口模式操作

1. 启动程序后，摄像头会自动打开
2. 将左手放在吉他指板上
3. 系统实时显示指法反馈：
   - **绿色**：正确按弦
   - **红色**：按错品位
   - **橙色**：多余手指
   - **白色**：未知状态
4. 按键控制：
   - `N` - 下一个和弦/音符
   - `P` - 上一个和弦/音符
   - `R` - 重置到开头
   - `Q` - 退出程序

### 最佳使用环境

- **光照**：保持充足均匀的光线
- **背景**：简洁的背景效果更好
- **距离**：手离摄像头 30-80cm
- **角度**：手掌正面朝向摄像头

### 界面说明

- **绿色圆点**：手部关键点（21个）
- **红色连线**：骨架连接
- **黄色标签**：左手/右手标识
- **彩色圆圈**：指尖位置反馈
- **左上角**：FPS 和检测手数
- **右上角**：当前歌曲和事件信息

---

## 🛠️ 技术栈

### 核心技术

- **OpenCV (cv2)**：视频处理和画面显示
- **MediaPipe**：Google 的手部关键点检测
- **NumPy**：数值计算

### MediaPipe 手部关键点

系统检测 21 个手部关键点：

```
0:  手腕 (Wrist)
1-4:  大拇指 (Thumb)
5-8:  食指 (Index)
9-12:  中指 (Middle)
13-16: 无名指 (Ring)
17-20: 小指 (Pinky)
```

---

## 🔧 配置选项

### 修改摄像头ID

如果默认摄像头无法使用，修改 `src/core/camera.py`：

```python
camera = Camera(camera_id=1)  # 尝试不同的ID：0, 1, 2
```

### 调整检测参数

在 `src/core/hand_tracker.py` 中：

```python
hand_tracker = HandTracker(
    max_hands=2,                    # 最大检测手数
    detection_confidence=0.5,       # 检测置信度
    tracking_confidence=0.5         # 追踪置信度
)
```

### 性能优化

降低分辨率以提高帧率，修改 `src/core/preprocessor.py`：

```python
preprocessor = VideoPreprocessor(
    flip=True,
    target_width=640  # 降低到 640（默认 1280）
)
```

---

## 🐛 常见问题

### Q: 摄像头打不开？

**A:** 检查是否有其他程序占用摄像头，或尝试更改 `camera_id`

### Q: 检测不到手？

**A:** 确保光线充足，手不要太近或太远，手掌正面朝向摄像头

### Q: FPS 太低（<15）？

**A:** 降低分辨率、减少最大检测手数，或关闭其他占用资源的程序

### Q: 找不到模型文件？

**A:** 确认 `assets/models/hand_landmarker.task` 文件存在且大小约 7.6MB

### Q: 导入 mediapipe 报错？

**A:** 确保安装的是最新版本：`pip install --upgrade mediapipe`

---

## 📅 开发路线图

### v0.1 ✅

- [x] 摄像头输入
- [x] 手部关键点检测
- [x] 骨架可视化
- [x] 基础架构搭建

### v0.2 ✅

- [x] 指板坐标映射
- [x] 标定系统
- [x] 和弦数据库
- [x] 基础反馈提示

### v0.3 ✅

- [x] 歌曲练习模式
- [x] 实时指法反馈
- [x] 代码重构优化
- [x] 模块化架构

### v0.5（当前版本）✅

- [x] Web UI 界面（Gradio）
- [x] 练习统计功能
- [x] 多首歌曲支持
- [x] 界面优化

### v1.0（计划中）

- [ ] 更多和弦支持
- [ ] 动作评分系统
- [ ] 进度追踪系统

### v2.0（长期目标）

- [ ] 机器学习动作识别
- [ ] 语音反馈
- [ ] 教学视频同步
- [ ] 进度追踪系统

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发建议

1. Fork 本项目
2. 创建新分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

---

## 📄 开源协议

本项目采用 MIT 协议，详见 [LICENSE](LICENSE) 文件

---

## 👨‍💻 作者

**Li Zeyu**

- 大学/学院：北京邮电大学/国际学院
- 联系方式：lzydavid123@gmail.com

---

## 🙏 致谢

- **MediaPipe**：提供强大的手部检测模型
- **OpenCV**：计算机视觉基础库
- 感谢所有为开源社区做出贡献的开发者

---

## 📊 项目状态

![GitHub last commit](https://img.shields.io/github/last-commit/notActiR/iGuitar)
![GitHub issues](https://img.shields.io/github/issues/notActiR/iGuitar)
![GitHub stars](https://img.shields.io/github/stars/notActiR/iGuitar)

---

**⭐ 如果这个项目对你有帮助，请给个 Star！**