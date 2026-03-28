"""坐标映射模块"""
import numpy as np
import cv2

class FretboardMapper:
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
        p = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(p, self.homography)
        fret, string = transformed[0, 0, 0], transformed[0, 0, 1]
        return fret, string

    def fretboard_to_pixel(self, fret, string):
        """将指板坐标 (fret, string) 映射回图像像素坐标 (x, y)"""
        p_fretboard = np.array([[fret, string]], dtype=np.float32).reshape(-1, 1, 2)
        H_inv = np.linalg.inv(self.homography)
        p_pixel_homog = cv2.perspectiveTransform(p_fretboard, H_inv)
        x, y = p_pixel_homog[0, 0, 0], p_pixel_homog[0, 0, 1]
        return int(x), int(y)

    def get_finger_frets(self, hand_landmarks, image_shape):
        h, w = image_shape[:2]
        result = {}
        for name, idx in self.FINGER_INDICES.items():
            lm = hand_landmarks[idx]
            x = int(lm.x * w)
            y = int(lm.y * h)
            fret, string = self.pixel_to_fretboard(x, y)
            fret_round = int(round(fret))
            string_round = int(round(string))
            if 0 <= fret_round <= 24 and 1 <= string_round <= 6:
                result[name] = (fret_round, string_round)
            else:
                result[name] = None
        return result