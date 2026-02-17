"""
识别器模块包
提供各种手写识别器的抽象和实现
"""

from .base import BaseRecognizer
from .chinese import ChineseHandwritingRecognizer
from .multi_char import MultiCharHandwritingRecognizer

__all__ = [
    "BaseRecognizer", 
    "ChineseHandwritingRecognizer",
    "MultiCharHandwritingRecognizer"
]