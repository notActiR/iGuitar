"""
视频预处理模块
职责：对原始帧进行预处理（翻转、缩放、色彩空间转换）
"""
import cv2


class VideoPreprocessor:
    def __init__(self, flip=True, target_width=None):
        """
        :param flip: 是否水平翻转（镜像效果，用户体验更好）
        :param target_width: 目标宽度，None则不缩放
        """
        self.flip = flip
        self.target_width = target_width

    def process(self, frame):
        """
        处理单帧图像
        :param frame: 原始BGR图像
        :return: 处理后的RGB图像
        """
        # 1. 水平翻转（镜像）
        if self.flip:
            frame = cv2.flip(frame, 1)

        # 2. 缩放（可选，提高处理速度）
        if self.target_width:
            h, w = frame.shape[:2]
            target_height = int(h * (self.target_width / w))
            frame = cv2.resize(frame, (self.target_width, target_height))

        # 3. BGR转RGB（MediaPipe需要）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return rgb_frame, frame  # 返回RGB（给MediaPipe）和BGR（给显示）