"""
UI显示模块
职责：在画面上绘制手部骨架和信息
"""
import cv2
import numpy as np


class Display:
    def __init__(self):
        # 手部连接关系（21个关键点的连接）
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # 大拇指
            (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
            (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
            (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
            (0, 17), (17, 18), (18, 19), (19, 20),  # 小指
            (5, 9), (9, 13), (13, 17)  # 手掌连接
        ]

        # 颜色配置
        self.landmark_color = (0, 255, 0)  # 绿色关键点
        self.connection_color = (255, 0, 0)  # 红色连接线
        self.text_color = (255, 255, 0)  # 黄色文字

    def draw_landmarks(self, frame, results):
        """
        在画面上绘制手部骨架（新版API）
        :param frame: BGR图像
        :param results: MediaPipe检测结果
        :return: 绘制后的图像
        """
        if not results.hand_landmarks:
            return frame

        h, w, _ = frame.shape

        for idx, hand_landmarks in enumerate(results.hand_landmarks):
            # 转换关键点为像素坐标
            points = []
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                points.append((cx, cy))

            # 绘制连接线
            for connection in self.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start_point = points[start_idx]
                end_point = points[end_idx]
                cv2.line(frame, start_point, end_point,
                        self.connection_color, 2)

            # 绘制关键点
            for point in points:
                cv2.circle(frame, point, 5, self.landmark_color, -1)
                cv2.circle(frame, point, 5, (255, 255, 255), 1)  # 白色边框

            # 显示左右手标签
            handedness = results.handedness[idx][0].category_name
            wrist = points[0]  # 手腕位置

            cv2.putText(frame, handedness,
                       (wrist[0] - 30, wrist[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       self.text_color, 2)

        return frame

    def add_info(self, frame, fps=0, hand_count=0):
        """
        添加信息文字
        :param frame: 图像
        :param fps: 帧率
        :param hand_count: 检测到的手数
        """
        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (250, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # 显示FPS
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 显示手数
        color = (0, 255, 0) if hand_count > 0 else (0, 0, 255)
        cv2.putText(frame, f'Hands: {hand_count}', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 显示API版本
        cv2.putText(frame, 'MediaPipe Tasks API', (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

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