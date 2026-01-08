"""
手部识别模块
职责：使用MediaPipe检测手部关键点
"""
import mediapipe as mp


class HandTracker:
    def __init__(self, max_hands=2, detection_confidence=0.7, tracking_confidence=0.5):
        """
        初始化手部追踪器
        :param max_hands: 最大检测手数
        :param detection_confidence: 检测置信度
        :param tracking_confidence: 追踪置信度
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

        print("✅ MediaPipe 手部追踪器初始化成功")

    def detect(self, rgb_frame):
        """
        检测手部关键点
        :param rgb_frame: RGB格式的图像
        :return: MediaPipe结果对象
        """
        results = self.hands.process(rgb_frame)
        return results

    def get_landmarks(self, results):
        """
        提取关键点坐标
        :param results: MediaPipe结果
        :return: 关键点列表 [{"hand": "Left/Right", "landmarks": [...]}]
        """
        if not results.multi_hand_landmarks:
            return []

        hands_data = []
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # 获取左右手信息
            handedness = results.multi_handedness[idx].classification[0].label

            # 提取21个关键点
            landmarks = []
            for lm in hand_landmarks.landmark:
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
        self.hands.close()