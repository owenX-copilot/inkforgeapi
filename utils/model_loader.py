"""
模型加载和注册工具
提供模型懒加载、注册和管理功能
"""

import yaml
import threading
from typing import Dict, Any, Optional, Type
from pathlib import Path
import logging

from recognizers.base import BaseRecognizer


class ModelRegistry:
    """
    模型注册表
    管理所有可用的识别器类型
    """

    _registry: Dict[str, Type[BaseRecognizer]] = {}

    @classmethod
    def register(cls, name: str, recognizer_class: Type[BaseRecognizer]) -> None:
        """
        注册识别器类

        Args:
            name: 识别器名称
            recognizer_class: 识别器类
        """
        cls._registry[name] = recognizer_class
        logging.info(f"注册识别器: {name} -> {recognizer_class.__name__}")

    @classmethod
    def get_recognizer_class(cls, name: str) -> Type[BaseRecognizer]:
        """
        获取识别器类

        Args:
            name: 识别器名称

        Returns:
            识别器类

        Raises:
            KeyError: 如果识别器未注册
        """
        if name not in cls._registry:
            raise KeyError(f"识别器 '{name}' 未注册。可用识别器: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def get_available_recognizers(cls) -> Dict[str, Type[BaseRecognizer]]:
        """
        获取所有可用的识别器

        Returns:
            识别器名称到类的映射
        """
        return cls._registry.copy()


class ModelLoader:
    """
    模型加载器
    负责模型的懒加载和管理
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化模型加载器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.models: Dict[str, BaseRecognizer] = {}
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()  # 添加线程锁

        # 加载配置
        self._load_config()

    def _load_config(self) -> None:
        """加载配置文件"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f) or {}
                self.logger.info(f"加载配置文件: {self.config_path}")
            else:
                self.logger.warning(f"配置文件不存在: {self.config_path}")
                self.config = {"models": {}}
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            self.config = {"models": {}}

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        获取模型配置

        Args:
            model_name: 模型名称

        Returns:
            模型配置字典
        """
        models_config = self.config.get("models", {})
        return models_config.get(model_name, {})

    def create_recognizer(self, model_name: str) -> BaseRecognizer:
        """
        创建识别器实例

        Args:
            model_name: 模型名称

        Returns:
            识别器实例

        Raises:
            ValueError: 如果模型配置无效
            KeyError: 如果识别器类型未注册
        """
        model_config = self.get_model_config(model_name)
        if not model_config:
            raise ValueError(f"模型 '{model_name}' 未在配置中定义")

        # 获取识别器类型
        recognizer_type = model_config.get("type")
        if not recognizer_type:
            raise ValueError(f"模型 '{model_name}' 未指定类型")

        # 获取识别器类
        try:
            recognizer_class = ModelRegistry.get_recognizer_class(recognizer_type)
        except KeyError as e:
            raise ValueError(f"无法创建识别器: {e}")

        # 获取模型路径
        model_path = model_config.get("model_path")
        if not model_path:
            raise ValueError(f"模型 '{model_name}' 未指定模型路径")

        # 确保路径是绝对路径
        model_path = Path(model_path)
        if not model_path.is_absolute():
            # 相对于配置文件所在目录
            model_path = self.config_path.parent / model_path

        # 创建识别器实例
        recognizer_config = model_config.get("config", {})
        recognizer = recognizer_class(str(model_path), recognizer_config)

        return recognizer

    def get_model(self, model_name: str, force_reload: bool = False) -> BaseRecognizer:
        """
        获取模型实例（线程安全）

        Args:
            model_name: 模型名称
            force_reload: 是否强制重新加载

        Returns:
            识别器实例

        Raises:
            ValueError: 如果模型配置无效
        """
        # 双重检查锁定模式
        if not force_reload and model_name in self.models:
            return self.models[model_name]
        
        with self._lock:
            # 再次检查，防止其他线程已经加载
            if not force_reload and model_name in self.models:
                return self.models[model_name]
            
            self.logger.info(f"加载模型: {model_name}")
            recognizer = self.create_recognizer(model_name)
            # 显式加载模型权重（不仅仅是创建实例）
            recognizer.load_model()
            recognizer.is_loaded = True
            self.models[model_name] = recognizer

        return self.models[model_name]

    def preload_models(self, model_names: Optional[list] = None) -> None:
        """
        预加载模型

        Args:
            model_names: 要预加载的模型名称列表，如果为 None 则加载所有模型
        """
        if model_names is None:
            model_names = list(self.config.get("models", {}).keys())

        for model_name in model_names:
            try:
                self.get_model(model_name)
                self.logger.info(f"预加载模型成功: {model_name}")
            except Exception as e:
                self.logger.error(f"预加载模型失败 {model_name}: {e}")

    def unload_model(self, model_name: str) -> None:
        """
        卸载模型

        Args:
            model_name: 模型名称
        """
        with self._lock:
            if model_name in self.models:
                del self.models[model_name]
                self.logger.info(f"卸载模型: {model_name}")

    def get_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """
        获取已加载模型的信息

        Returns:
            模型名称到信息的映射
        """
        result = {}
        for name, recognizer in self.models.items():
            try:
                result[name] = recognizer.get_info()
            except Exception as e:
                result[name] = {"error": str(e)}
        return result

    def reload_config(self) -> None:
        """重新加载配置文件"""
        old_config = self.config.copy()
        self._load_config()

        # 检查配置变化
        old_models = set(old_config.get("models", {}).keys())
        new_models = set(self.config.get("models", {}).keys())

        # 卸载已删除的模型
        for model_name in old_models - new_models:
            if model_name in self.models:
                self.unload_model(model_name)

        self.logger.info("配置文件已重新加载")
