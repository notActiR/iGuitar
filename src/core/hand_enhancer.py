# src/core/hand_enhancer.py
"""
论文启发的手部姿态后处理增强模块
- 时序平滑 (基于 CVPR 2025)
- 几何约束 (基于 TCYB 2021)
"""
import numpy as np
from collections import deque

class HandEnhancer:
    def __init__(self, use_temporal=True, use_geometric=True, history_len=5):
        self.use_temporal = use_temporal
        self.use_geometric = use_geometric
        self.history = deque(maxlen=history_len)  # 存储每帧所有关键点坐标

    def apply(self, hand_landmarks, image_shape):
        """
        对 MediaPipe 检测到的关键点进行后处理
        hand_landmarks: 可迭代对象，每个元素有 x, y 属性
        image_shape: (h, w) 用于归一化坐标转换（此处直接操作归一化坐标）
        返回修改后的 hand_landmarks（原地修改）
        """
        if not self.use_temporal and not self.use_geometric:
            return hand_landmarks

        # 提取所有关键点的 (x, y) 归一化坐标
        points = np.array([[lm.x, lm.y] for lm in hand_landmarks])

        # 1. 时序平滑（指数移动平均）
        if self.use_temporal:
            self.history.append(points.copy())
            if len(self.history) > 1:
                alpha = 0.7  # 当前帧权重
                smoothed = alpha * points + (1 - alpha) * np.mean(self.history, axis=0)
                points = smoothed

        # 2. 几何约束：限制指尖与手腕的距离（防止飞点）
        if self.use_geometric:
            wrist = points[0]  # 手腕点索引0
            for idx in [4, 8, 12, 16, 20]:  # 指尖点
                dist = np.linalg.norm(points[idx] - wrist)
                if dist > 0.5:  # 归一化距离阈值（经验值）
                    # 拉回合理范围
                    direction = (points[idx] - wrist) / (dist + 1e-6)
                    points[idx] = wrist + direction * 0.5

        # 写回原对象
        for i, lm in enumerate(hand_landmarks):
            lm.x = float(points[i, 0])
            lm.y = float(points[i, 1])
        return hand_landmarks