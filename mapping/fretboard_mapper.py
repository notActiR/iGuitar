"""
坐标映射模块：利用标定矩阵将指尖像素坐标转换为指板上的品和弦
"""
import numpy as np
import cv2

class FretboardMapper:
    # MediaPipe 指尖关键点索引
    FINGER_INDICES = {
        'thumb': 4,
        'index': 8,
        'middle': 12,
        'ring': 16,
        'little': 20
    }

    def __init__(self, matrix_path='calibration_matrix.npy'):
        self.homography = np.load(matrix_path)

    def pixel_to_fretboard(self, x, y):
        """转换单个点"""
        p = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(p, self.homography)
        fret, string = transformed[0, 0, 0], transformed[0, 0, 1]
        return fret, string

    def get_finger_frets(self, hand_landmarks, image_shape):
        """
        从 MediaPipe 手部关键点提取各指尖的指板坐标
        :param hand_landmarks: MediaPipe 检测到的单手 landmark 列表（21个点）
        :param image_shape: 图像形状 (h, w, c)
        :return: 字典 {finger_name: (fret, string)}，若手指不在指板范围内则为 None
        """
        h, w = image_shape[:2]
        result = {}
        for name, idx in self.FINGER_INDICES.items():
            lm = hand_landmarks[idx]
            x = int(lm.x * w)
            y = int(lm.y * h)
            fret, string = self.pixel_to_fretboard(x, y)
            # 四舍五入取整，并检查是否在合理范围内（品 0~24，弦 1~6）
            fret_round = int(round(fret))
            string_round = int(round(string))
            if 0 <= fret_round <= 24 and 1 <= string_round <= 6:
                result[name] = (fret_round, string_round)
            else:
                result[name] = None   # 手指不在指板上（例如悬空或超出）
        return result