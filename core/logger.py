# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

# core/logger.py - ForgeX v2 日誌系統（修復 basicConfig 多次調用問題）
import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional, Callable

from core.config import LOGS_DIR


def setup_logger(name: str = "ForgeX", level: int = logging.INFO) -> logging.Logger:
    """創建獨立 logger（不依賴 basicConfig）"""
    logger = logging.getLogger(name)

    # 避免重複添加 handler
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # 控制台
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # 文件（輪轉，避免長期運行生成超大文件）
    log_file = LOGS_DIR / f"{name}_{datetime.now():%Y%m%d}.log"
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# 主 logger
_logger = setup_logger("ForgeX")


def log(msg: str, level: str = "info"):
    """統一日誌函數"""
    getattr(_logger, level, _logger.info)(msg)


def get_log_callback(name: str = "ForgeX") -> Callable[[str], None]:
    """獲取日誌回調函數"""
    logger = setup_logger(name)
    return lambda msg: logger.info(msg)
