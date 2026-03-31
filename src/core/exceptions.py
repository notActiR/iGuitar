"""自定义异常类"""

class iGuitarException(Exception):
    """基础异常类"""
    pass

class CameraError(iGuitarException):
    """摄像头相关错误"""
    pass

class CalibrationError(iGuitarException):
    """标定相关错误"""
    pass

class ModelError(iGuitarException):
    """模型加载错误"""
    pass

class ConfigError(iGuitarException):
    """配置错误"""
    pass
