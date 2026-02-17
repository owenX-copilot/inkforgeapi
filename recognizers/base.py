"""
识别器抽象基类
为不同手写识别模型提供统一接口
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import torch
import numpy as np


class BaseRecognizer(ABC):
    """手写识别器抽象基类"""

    def __init__(self, model_path: str, config: Optional[Dict[str, Any]] = None):
        """
        初始化识别器

        Args:
            model_path: 模型文件路径
            config: 配置字典
        """
        self.model_path = model_path
        self.config = config or {}
        self.model = None
        self.device = torch.device("cpu")
        self.is_loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """
        加载模型
        子类必须实现此方法
        """
        pass

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        图像预处理
        子类必须实现此方法

        Args:
            image: 输入图像 (H, W) 或 (H, W, C)

        Returns:
            预处理后的张量
        """
        pass

    @abstractmethod
    def predict(self, image: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        执行预测

        Args:
            image: 输入图像
            top_k: 返回前K个预测结果

        Returns:
            预测结果列表，每个元素包含字符、置信度、索引等信息
        """
        pass

    @abstractmethod
    def get_class_mapping(self) -> Dict[int, str]:
        """
        获取类别映射

        Returns:
            索引到字符的映射字典
        """
        pass

    def lazy_load(self) -> None:
        """懒加载模型（首次使用时加载）"""
        if not self.is_loaded:
            self.load_model()
            self.is_loaded = True

    def get_info(self) -> Dict[str, Any]:
        """
        获取识别器信息

        Returns:
            包含识别器信息的字典
        """
        return {
            "name": self.__class__.__name__,
            "model_path": self.model_path,
            "is_loaded": self.is_loaded,
            "device": str(self.device),
            "config": self.config
        }

    def __del__(self):
        """清理资源"""
        if self.model is not None:
            del self.model