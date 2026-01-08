"""
UI显示模块
职责：在画面上绘制手部骨架和信息
"""
import cv2
import mediapipe as mp


class Display:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        # 自定义骨架样式
        self.drawing_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0),  # 绿色
            thickness=2,
            circle_radius=3
        )

        self.connection_spec = self.mp_drawing.DrawingSpec(
            color=(255, 0, 0),  # 红色
            thickness=2
        )

    def draw_landmarks(self, frame, results):
        """
        在画面上绘制手部骨架
        :param frame: BGR图像
        :param results: MediaPipe检测结果
        :return: 绘制后的图像
        """
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # 绘制关键点和连接线
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.drawing_spec,
                    self.connection_spec
                )

                # 显示左右手标签
                handedness = results.multi_handedness[hand_idx].classification[0].label
                h, w, _ = frame.shape

                # 在手腕位置显示标签
                wrist = hand_landmarks.landmark[0]
                cx, cy = int(wrist.x * w), int(wrist.y * h)

                cv2.putText(frame, handedness, (cx - 30, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return frame

    def add_info(self, frame, fps=0, hand_count=0):
        """
        添加信息文字
        :param frame: 图像
        :param fps: 帧率
        :param hand_count: 检测到的手数
        """
        # 显示FPS
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 显示手数
        cv2.putText(frame, f'Hands: {hand_count}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 显示退出提示
        cv2.putText(frame, 'Press Q to quit', (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def show(self, window_name, frame):
        """
        显示窗口
        :param window_name: 窗口名称
        :param frame: 要显示的图像
        """
        cv2.imshow(window_name, frame)

    @staticmethod
    def wait_key(delay=1):
        """
        等待按键
        :param delay: 延迟时间（毫秒）
        :return: 按键值
        """
        return cv2.waitKey(delay) & 0xFF

    @staticmethod
    def destroy_all():
        """关闭所有窗口"""
        cv2.destroyAllWindows()