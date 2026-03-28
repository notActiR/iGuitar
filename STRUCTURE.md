# iGuitar - 项目结构说明

## 📁 新的文件结构

```
iGuitar/
├── src/                    # 源代码
│   ├── core/              # 核心功能模块
│   │   ├── camera.py      # 摄像头
│   │   ├── hand_tracker.py # 手部追踪
│   │   └── preprocessor.py # 视频预处理
│   ├── mapping/           # 坐标映射
│   │   ├── calibrator.py  # 标定器
│   │   └── fretboard_mapper.py # 指板映射
│   ├── data/              # 数据模块
│   │   ├── chord_db.py    # 和弦数据库
│   │   └── song.py        # 歌曲管理
│   └── ui/                # UI显示
│       └── display.py     # 显示模块
├── scripts/               # 可执行脚本
│   ├── main.py           # 主程序
│   └── calibrate.py      # 标定脚本
├── assets/               # 资源文件
│   ├── models/           # ML模型
│   └── songs/            # 歌曲文件
└── requirements.txt      # 依赖列表
```

## 🚀 使用方法

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行标定
```bash
python scripts/calibrate.py
```

### 3. 运行主程序
```bash
python scripts/main.py
```

## ✨ 优化内容

1. **删除重复代码**：移除了重复的 finger_indices 字典
2. **删除未使用函数**：camera.get_fps() 和 chord_db.check_chord()
3. **简化注释**：删除冗余文档字符串
4. **优化代码结构**：使用字典映射替代 if-else 链
5. **重组文件架构**：更清晰的模块化结构
