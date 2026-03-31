# src/mapping/fretboard_mapper.py
import numpy as np
import cv2
from collections import deque

class FretboardMapper:
    FINGER_INDICES = {
        'thumb': 4,
        'index': 8,
        'middle': 12,
        'ring': 16,
        'little': 20
    }

    def __init__(self, matrix_path='calibration_matrix.npy', use_smoothing=True, filter_size=5):
        self.homography = np.load(matrix_path)
        self.use_smoothing = use_smoothing
        self.filter_size = filter_size
        self.point_filters = {name: deque(maxlen=filter_size) for name in self.FINGER_INDICES.keys()}
        self.distortion_coeffs = None
        self.camera_matrix = None

    def set_distortion_params(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = dist_coeffs

    def undistort_point(self, x, y):
        if self.camera_matrix is not None and self.distortion_coeffs is not None:
            src_pts = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
            dst_pts = cv2.undistortPoints(src_pts, self.camera_matrix, self.distortion_coeffs, P=self.camera_matrix)
            return int(dst_pts[0, 0, 0]), int(dst_pts[0, 0, 1])
        return x, y

    def smooth_point(self, finger, x, y):
        if not self.use_smoothing:
            return x, y
        q = self.point_filters[finger]
        q.append((x, y))
        avg_x = int(sum(p[0] for p in q) / len(q))
        avg_y = int(sum(p[1] for p in q) / len(q))
        return avg_x, avg_y

    def pixel_to_fretboard(self, x, y):
        p = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(p, self.homography)
        fret, string = transformed[0, 0, 0], transformed[0, 0, 1]
        return fret, string

    def fretboard_to_pixel(self, fret, string):
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
            # 修复置信度过滤：检查 visibility 是否为 None
            if hasattr(lm, 'visibility') and lm.visibility is not None and lm.visibility < 0.5:
                result[name] = None
                continue

            x_raw = int(lm.x * w)
            y_raw = int(lm.y * h)
            x_corr, y_corr = self.undistort_point(x_raw, y_raw)
            x_smooth, y_smooth = self.smooth_point(name, x_corr, y_corr)
            fret, string = self.pixel_to_fretboard(x_smooth, y_smooth)
            fret_round = int(round(fret))
            string_round = int(round(string))
            if 0 <= fret_round <= 24 and 1 <= string_round <= 6:
                result[name] = (fret_round, string_round)
            else:
                result[name] = None
        return result