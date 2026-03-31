"""摄像头输入模块"""
import cv2
from .logger import setup_logger
from .exceptions import CameraError

logger = setup_logger(__name__)

class Camera:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)

        if not self.cap.isOpened():
            raise CameraError(
                f"无法打开摄像头 ID={camera_id}\n"
                "请检查：1) 摄像头是否连接 2) 是否被其他程序占用 3) 尝试其他 ID"
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        logger.info(f"摄像头初始化成功 (ID={camera_id})")

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("读取帧失败")
        return ret, frame

    def release(self):
        self.cap.release()
        logger.info("摄像头已关闭")