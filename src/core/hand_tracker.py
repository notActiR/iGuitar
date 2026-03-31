# src/core/hand_tracker.py
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .hand_enhancer import HandEnhancer

class HandTracker:
    def __init__(self, model_path='assets/models/hand_landmarker.task',
                 max_hands=2,
                 detection_confidence=0.5,
                 tracking_confidence=0.5,
                 use_enhancer=False):
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
        self.use_enhancer = use_enhancer
        if self.use_enhancer:
            self.enhancer = HandEnhancer()
        else:
            self.enhancer = None
        print("✅ MediaPipe HandTracker initialized")

    def detect(self, rgb_frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.frame_timestamp_ms += 33
        results = self.detector.detect_for_video(mp_image, self.frame_timestamp_ms)

        if self.use_enhancer and results.hand_landmarks:
            for hand_lms in results.hand_landmarks:
                self.enhancer.apply(hand_lms, rgb_frame.shape)
        return results

    def get_landmarks(self, results):
        if not results.hand_landmarks:
            return []
        hands_data = []
        for idx, hand_landmarks in enumerate(results.hand_landmarks):
            handedness = results.handedness[idx][0].category_name
            landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in hand_landmarks]
            hands_data.append({'hand': handedness, 'landmarks': landmarks})
        return hands_data

    def close(self):
        self.detector.close()
        print("🔒 HandTracker closed")