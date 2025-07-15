import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

# 日志目录
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# 日志文件名精确到分钟
log_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
log_file = os.path.join(LOG_DIR, f"{log_time}.log")

# 获取全局 logger
logger = logging.getLogger('fall_detection')
logger.setLevel(logging.DEBUG)

# 防止重复添加 handler
if not logger.handlers:
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 文件输出（滚动文件大小5MB，保留5个备份）
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # 加入 handler
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
