"""
标定模块：负责采集标定点、计算透视变换矩阵并保存/加载
"""
import cv2
import numpy as np
import json
import os

class Calibrator:
    def __init__(self):
        self.src_points = []   # 图像坐标 [(x1,y1), ...]
        self.dst_points = []   # 指板坐标 [(fret, string), ...]
        self.homography = None

    def add_point(self, pixel_x, pixel_y, fret, string):
        """添加一个标定点"""
        self.src_points.append([pixel_x, pixel_y])
        self.dst_points.append([fret, string])

    def clear(self):
        self.src_points = []
        self.dst_points = []
        self.homography = None

    def compute_homography(self):
        """计算透视变换矩阵（单应性矩阵）"""
        if len(self.src_points) < 4:
            raise ValueError("至少需要4个点")
        src = np.float32(self.src_points)
        dst = np.float32(self.dst_points)
        # 使用 findHomography 获得更鲁棒的结果（RANSAC自动剔除误差大的点）
        self.homography, mask = cv2.findHomography(src, dst, method=cv2.RANSAC)
        return self.homography

    def save(self, path='calibration_matrix.npy'):
        """保存矩阵"""
        np.save(path, self.homography)
        # 可选：保存点信息以便调试
        with open('calibration_points.json', 'w') as f:
            json.dump({'src': self.src_points, 'dst': self.dst_points}, f)

    def load(self, path='calibration_matrix.npy'):
        """加载矩阵"""
        self.homography = np.load(path)
        return self.homography

    def pixel_to_fretboard(self, pixel_x, pixel_y):
        """将像素坐标转换为指板坐标 (fret, string)"""
        if self.homography is None:
            raise RuntimeError("未标定或加载矩阵")
        p = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(p, self.homography)
        fret, string = transformed[0, 0, 0], transformed[0, 0, 1]
        return fret, string