"""手部识别模块"""
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from .logger import setup_logger
from .exceptions import ModelError

logger = setup_logger(__name__)

class HandTracker:
    def __init__(self, model_path='assets/models/hand_landmarker.task',
                 max_hands=2,
                 detection_confidence=0.5,
                 tracking_confidence=0.5):
        try:
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=detection_confidence,
                min_hand_presence_confidence=tracking_confidence,
                min_tracking_confidence=tracking_confidence
            )

            self.detector = vision.HandLandmarker.create_from_options(options)
            self.frame_timestamp_ms = 0

            logger.info("MediaPipe 手部追踪器初始化成功")
        except Exception as e:
            raise ModelError(f"模型加载失败: {e}\n请确认模型文件存在: {model_path}")

    def detect(self, rgb_frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.frame_timestamp_ms += 33
        results = self.detector.detect_for_video(mp_image, self.frame_timestamp_ms)
        return results

    def get_landmarks(self, results):
        if not results.hand_landmarks:
            return []

        hands_data = []

        for idx, hand_landmarks in enumerate(results.hand_landmarks):
            handedness = results.handedness[idx][0].category_name

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
        self.detector.close()
        logger.info("手部追踪器已关闭")