"""常量定义"""

# 手部关键点索引
FINGER_TIPS = {
    'thumb': 4,
    'index': 8,
    'middle': 12,
    'ring': 16,
    'little': 20
}

# 颜色定义 (BGR)
COLORS = {
    'correct': (0, 255, 0),      # 绿色
    'wrong': (0, 0, 255),        # 红色
    'extra': (0, 165, 255),      # 橙色
    'missing': (128, 128, 128),  # 灰色
    'skeleton': (255, 0, 0),     # 蓝色
    'landmark': (0, 255, 0)      # 绿色
}

# 默认阈值
THRESHOLDS = {
    'distance': 30,              # 像素距离阈值
    'detection_confidence': 0.5,
    'tracking_confidence': 0.5
}

# 指板参数
FRETBOARD = {
    'strings': 6,
    'frets': 12,
    'max_frets': 24
}
