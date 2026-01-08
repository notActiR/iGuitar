# 🎸 iGuitar - 吉他学习辅助系统

> 基于计算机视觉的吉他教学辅助工具，使用 MediaPipe 实时检测手部动作，帮助初学者纠正指法

![Version](https://img.shields.io/badge/version-0.1-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![MediaPipe](https://img.shields.io/badge/mediapipe-latest-orange)

---

## 📖 项目简介

iGuitar 是一个面向吉他初学者的智能学习辅助系统。通过摄像头实时捕捉用户的手部动作，与标准指法进行对比，并给出即时反馈，帮助学习者快速掌握正确的弹奏姿势。

### 核心功能（v0.1）

- ✅ 实时手部关键点检测（21个关键点）
- ✅ 双手同时追踪
- ✅ 骨架可视化显示
- ✅ FPS 性能监控
- 🚧 动作标准对比（开发中）
- 🚧 智能反馈系统（计划中）

---

## 🏗️ 系统架构

```
┌──────────────┐
│   摄像头输入   │  Camera Module
└──────┬───────┘
       ↓
┌──────────────┐
│  视频预处理   │  Preprocessing (翻转/缩放/色彩转换)
└──────┬───────┘
       ↓
┌──────────────┐
│  手部识别    │  MediaPipe Hand Tracking
└──────┬───────┘
       ↓
┌──────────────┐
│  特征提取    │  Feature Extraction
└──────┬───────┘
       ↓
┌──────────────┐
│  动作评估    │  Evaluation (规则/ML)
└──────┬───────┘
       ↓
┌──────────────┐
│  反馈生成    │  Feedback Generator
└──────┬───────┘
       ↓
┌──────────────┐
│   用户界面   │  UI Display
└──────────────┘
```

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
```

4. **下载模型文件**

创建 `models` 目录并下载 MediaPipe 手部检测模型：

```bash
mkdir models
cd models

# Windows PowerShell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "hand_landmarker.task"

# Mac/Linux
curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

cd ..
```

或手动下载：[下载链接](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)

将文件放到 `iGuitar/models/hand_landmarker.task`

5. **运行程序**

```bash
python main.py
```

---

## 📂 项目结构

```
iGuitar/
│
├── main.py                      # 程序入口
│
├── camera/                      # 摄像头模块
│   └── camera.py                # 摄像头输入控制
│
├── vision/                      # 视觉处理模块
│   ├── __init__.py
│   ├── preprocess.py            # 视频预处理
│   └── hand_tracker.py          # 手部追踪（MediaPipe）
│
├── feature/                     # 特征提取模块（待开发）
│   └── extractor.py
│
├── evaluation/                  # 动作评估模块（待开发）
│   └── chord_evaluator.py
│
├── feedback/                    # 反馈生成模块（待开发）
│   └── feedback_generator.py
│
├── ui/                          # 用户界面模块
│   └── display.py               # 画面显示和标注
│
├── data/                        # 数据存储
│   └── chords/                  # 标准和弦数据
│       └── C.json               # C大调和弦（示例）
│
├── models/                      # 模型文件
│   └── hand_landmarker.task     # MediaPipe 手部模型
│
├── utils/                       # 工具函数
│   └── __init__.py
│
├── requirements.txt             # 依赖列表
└── README.md                    # 项目文档
```

---

## 🎮 使用说明

### 基本操作

1. 启动程序后，摄像头会自动打开
2. 将双手放在摄像头前（距离 30-80cm 最佳）
3. 系统会实时显示手部骨架和关键点
4. 按 `Q` 键退出程序

### 最佳使用环境

- **光照**：保持充足均匀的光线
- **背景**：简洁的背景效果更好
- **距离**：手离摄像头 30-80cm
- **角度**：手掌正面朝向摄像头

### 界面说明

- **绿色圆点**：手部关键点（21个）
- **红色连线**：骨架连接
- **黄色标签**：左手/右手标识
- **左上角**：FPS 和检测手数

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

如果默认摄像头无法使用，修改 `camera/camera.py`：

```python
camera = Camera(camera_id=1)  # 尝试不同的ID：0, 1, 2
```

### 调整检测参数

在 `vision/hand_tracker.py` 中：

```python
hand_tracker = HandTracker(
    max_hands=2,                    # 最大检测手数
    detection_confidence=0.5,       # 检测置信度
    tracking_confidence=0.5         # 追踪置信度
)
```

### 性能优化

降低分辨率以提高帧率，修改 `vision/preprocess.py`：

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

**A:** 确认 `models/hand_landmarker.task` 文件存在且大小约 6MB

### Q: 导入 mediapipe 报错？

**A:** 确保安装的是最新版本：`pip install --upgrade mediapipe`

---

## 📅 开发路线图

### v0.1（当前版本）✅

- [x] 摄像头输入
- [x] 手部关键点检测
- [x] 骨架可视化
- [x] 基础架构搭建

### v0.2（下一版本）

- [ ] 录制标准和弦动作
- [ ] 保存关键点数据
- [ ] 简单的动作对比
- [ ] 基础反馈提示

### v0.3（计划中）

- [ ] 多和弦支持
- [ ] 动作评分系统
- [ ] 详细反馈文字
- [ ] 数据统计功能

### v1.0（长期目标）

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

**你的名字**

- 大学/学院：XX大学
- 年级专业：大二 计算机科学
- 联系方式：your.email@example.com

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