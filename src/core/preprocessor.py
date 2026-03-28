"""视频预处理模块"""
import cv2


class VideoPreprocessor:
    def __init__(self, flip=True, target_width=None):
        self.flip = flip
        self.target_width = target_width

    def process(self, frame):
        if self.flip:
            frame = cv2.flip(frame, 1)

        if self.target_width:
            h, w = frame.shape[:2]
            target_height = int(h * (self.target_width / w))
            frame = cv2.resize(frame, (self.target_width, target_height))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb_frame, frame