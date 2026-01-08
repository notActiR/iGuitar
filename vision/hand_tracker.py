"""
手部识别模块
职责：使用MediaPipe Tasks API检测手部关键点
"""
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np


class HandTracker:
    def __init__(self, model_path='models/hand_landmarker.task',
                 max_hands=2,
                 detection_confidence=0.5,
                 tracking_confidence=0.5):
        """
        初始化手部追踪器
        :param model_path: 模型文件路径
        :param max_hands: 最大检测手数
        :param detection_confidence: 检测置信度
        :param tracking_confidence: 追踪置信度
        """
        # 配置选项
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,  # 视频模式
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=tracking_confidence,
            min_tracking_confidence=tracking_confidence
        )

        # 创建检测器
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.frame_timestamp_ms = 0

        print("✅ MediaPipe 手部追踪器初始化成功 (新版API)")

    def detect(self, rgb_frame):
        """
        检测手部关键点
        :param rgb_frame: RGB格式的图像（numpy array）
        :return: 检测结果
        """
        # 转换为 MediaPipe Image 格式
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 检测（需要时间戳）
        self.frame_timestamp_ms += 33  # 约30fps
        results = self.detector.detect_for_video(mp_image, self.frame_timestamp_ms)

        return results

    def get_landmarks(self, results):
        """
        提取关键点坐标
        :param results: MediaPipe结果
        :return: 关键点列表 [{"hand": "Left/Right", "landmarks": [...]}]
        """
        if not results.hand_landmarks:
            return []

        hands_data = []

        # 新版API的数据结构
        for idx, hand_landmarks in enumerate(results.hand_landmarks):
            # 获取左右手信息
            handedness = results.handedness[idx][0].category_name

            # 提取21个关键点
            landmarks = []
            for lm in hand_landmarks:
                landmarks.append({
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z
                })

            hands_data.append({
                'hand': handedness,
                'landmarks': landmarks
            })

        return hands_data

    def close(self):
        """释放资源"""
        self.detector.close()
        print("🔒 手部追踪器已关闭")