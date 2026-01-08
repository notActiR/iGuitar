"""
摄像头输入模块
职责：从摄像头读取视频帧
"""
import cv2


class Camera:
    def __init__(self, camera_id=0):
        """
        初始化摄像头
        :param camera_id: 摄像头ID，默认0（笔记本内置摄像头）
        """
        self.cap = cv2.VideoCapture(camera_id)

        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头，请检查摄像头连接")

        # 设置摄像头分辨率（可选）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("✅ 摄像头初始化成功")

    def read_frame(self):
        """
        读取一帧图像
        :return: (success, frame)
        """
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        """释放摄像头资源"""
        self.cap.release()
        print("📷 摄像头已关闭")

    def get_fps(self):
        """获取摄像头FPS"""
        return self.cap.get(cv2.CAP_PROP_FPS)