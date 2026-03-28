"""UI显示模块"""
import cv2
import numpy as np


class Display:
    def __init__(self):
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]

        self.landmark_color = (0, 255, 0)
        self.connection_color = (255, 0, 0)
        self.text_color = (255, 255, 0)

    def draw_landmarks(self, frame, results):
        if not results.hand_landmarks:
            return frame

        h, w, _ = frame.shape

        for idx, hand_landmarks in enumerate(results.hand_landmarks):
            points = []
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                points.append((cx, cy))

            for connection in self.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start_point = points[start_idx]
                end_point = points[end_idx]
                cv2.line(frame, start_point, end_point,
                        self.connection_color, 2)

            for point in points:
                cv2.circle(frame, point, 5, self.landmark_color, -1)
                cv2.circle(frame, point, 5, (255, 255, 255), 1)

            handedness = results.handedness[idx][0].category_name
            wrist = points[0]

            cv2.putText(frame, handedness,
                       (wrist[0] - 30, wrist[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       self.text_color, 2)

        return frame

    def add_info(self, frame, fps=0, hand_count=0):
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (250, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        color = (0, 255, 0) if hand_count > 0 else (0, 0, 255)
        cv2.putText(frame, f'Hands: {hand_count}', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame, 'MediaPipe Tasks API', (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.putText(frame, 'Press Q to quit', (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def show(self, window_name, frame):
        cv2.imshow(window_name, frame)

    @staticmethod
    def wait_key(delay=1):
        return cv2.waitKey(delay) & 0xFF

    @staticmethod
    def destroy_all():
        cv2.destroyAllWindows()