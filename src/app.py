"""应用主类 - 统一管理核心组件"""
from .core import Camera, HandTracker, VideoPreprocessor
from .core.config import Config
from .core.logger import setup_logger
from .core.exceptions import iGuitarException
from .mapping import FretboardMapper
from .ui.display import Display

logger = setup_logger(__name__)

class iGuitarApp:
    """iGuitar 应用主类"""

    def __init__(self, config_path='config.json'):
        self.config = Config(config_path)
        self.camera = None
        self.hand_tracker = None
        self.preprocessor = None
        self.mapper = None
        self.display = None

    def initialize(self):
        """初始化所有组件"""
        try:
            logger.info("开始初始化 iGuitar 应用...")

            # 摄像头
            camera_id = self.config.get('camera', 'id', default=0)
            self.camera = Camera(camera_id)

            # 手部追踪
            self.hand_tracker = HandTracker(
                max_hands=self.config.get('hand_tracker', 'max_hands', default=2),
                detection_confidence=self.config.get('hand_tracker', 'detection_confidence', default=0.5),
                tracking_confidence=self.config.get('hand_tracker', 'tracking_confidence', default=0.5)
            )

            # 预处理器
            self.preprocessor = VideoPreprocessor(
                flip=self.config.get('display', 'flip_horizontal', default=True)
            )

            # 指板映射
            self.mapper = FretboardMapper()

            # 显示器
            self.display = Display()

            logger.info("iGuitar 应用初始化完成")
            return True

        except iGuitarException as e:
            logger.error(f"初始化失败: {e}")
            return False

    def cleanup(self):
        """清理资源"""
        if self.camera:
            self.camera.release()
        if self.hand_tracker:
            self.hand_tracker.close()
        logger.info("应用已清理")
