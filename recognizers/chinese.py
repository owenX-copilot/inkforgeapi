"""
中文手写识别器
基于 HanziTiny 模型实现
"""

import json
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# 添加模型目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from model.hanzi_tiny import HanziTiny
from .base import BaseRecognizer


class ChineseHandwritingRecognizer(BaseRecognizer):
    """中文手写识别器"""

    def __init__(self, model_path: str, config: Optional[Dict[str, Any]] = None):
        """
        初始化中文手写识别器

        Args:
            model_path: 模型文件路径
            config: 配置字典，可包含以下键：
                - class_mapping_path: 类别映射文件路径
                - img_size: 图像尺寸 (默认64)
                - mean: 标准化均值 (默认0.5)
                - std: 标准化标准差 (默认0.5)
        """
        super().__init__(model_path, config)

        # 默认配置
        self.config.setdefault("class_mapping_path",
                              str(Path(model_path).parent / "classes.json"))
        self.config.setdefault("img_size", 64)
        self.config.setdefault("mean", 0.5)
        self.config.setdefault("std", 0.5)

        self.img_size = self.config["img_size"]
        self.mean = self.config["mean"]
        self.std = self.config["std"]

        # 类别映射
        self.idx_to_char = {}
        self.char_to_idx = {}

        # 加载类别映射
        self._load_class_mapping()

    def _load_class_mapping(self) -> None:
        """加载类别映射"""
        mapping_path = self.config["class_mapping_path"]
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                encoded_classes = json.load(f)

            # 解码 Unicode 编码 (#U4e00 -> '一')
            for idx, encoded_char in enumerate(encoded_classes):
                if encoded_char.startswith("#U"):
                    # 提取十六进制编码并转换为字符
                    hex_code = encoded_char[2:]  # 去掉 "#U"
                    char = chr(int(hex_code, 16))
                else:
                    char = encoded_char

                self.idx_to_char[idx] = char
                self.char_to_idx[char] = idx

            print(f"加载类别映射成功，共 {len(self.idx_to_char)} 个字符")

        except Exception as e:
            print(f"加载类别映射失败: {e}")
            # 创建默认映射（0-629）
            for i in range(630):
                self.idx_to_char[i] = f"char_{i}"
                self.char_to_idx[f"char_{i}"] = i

    def load_model(self) -> None:
        """加载模型"""
        try:
            # 加载模型定义
            num_classes = len(self.idx_to_char)
            self.model = HanziTiny(num_classes=num_classes)

            # 加载权重
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # 检查检查点格式
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    # 假设整个字典就是状态字典
                    self.model.load_state_dict(checkpoint)
            else:
                # 假设是直接的状态字典
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True

            print(f"模型加载成功: {self.model_path}")
            print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"模型设备: {next(self.model.parameters()).device}")
            print(f"模型训练模式: {self.model.training}")
            print(f"模型类别数: {self.model.num_classes if hasattr(self.model, 'num_classes') else '未知'}")

        except Exception as e:
            print(f"模型加载失败: {e}")
            raise

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        图像预处理

        Args:
            image: 输入图像，可以是 (H, W) 或 (H, W, C)

        Returns:
            预处理后的张量 [1, 1, img_size, img_size]
        """
        import cv2
        from PIL import Image, ImageOps

        # 确保是 numpy 数组
        if not isinstance(image, np.ndarray):
            raise ValueError("输入必须是 numpy 数组")

        # 转换为灰度图
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            elif image.shape[2] == 1:
                image = image.squeeze()

        # 基础切割：去掉多余的留白
        # 将numpy数组转换为PIL图像进行裁剪
        pil_img = Image.fromarray(image, mode='L')

        # 反色以便找到笔迹边界（白底灰字 -> 黑底亮灰字）
        inverted = ImageOps.invert(pil_img)
        bbox = inverted.getbbox()  # 找到非黑色像素的边界

        if bbox:
            # 找到边界后，稍微往外扩一点 (Padding)，避免字撑得太满贴边
            left, upper, right, lower = bbox
            p = 10  # Padding 像素
            left = max(0, left - p)
            upper = max(0, upper - p)
            right = min(pil_img.width, right + p)
            lower = min(pil_img.height, lower + p)

            # 裁剪出来
            pil_img = pil_img.crop((left, upper, right, lower))
            image = np.array(pil_img)

        # 调整尺寸
        if image.shape != (self.img_size, self.img_size):
            image = cv2.resize(image, (self.img_size, self.img_size),
                              interpolation=cv2.INTER_AREA)

        # 转换为 [0, 1] 范围
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0

        # 标准化
        image = (image - self.mean) / self.std

        # 添加批次和通道维度 [1, 1, H, W]
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

        return image_tensor

    def predict(self, image: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        执行预测

        Args:
            image: 输入图像
            top_k: 返回前K个预测结果

        Returns:
            预测结果列表，每个元素包含:
                - character: 字符
                - confidence: 置信度
                - index: 类别索引
                - unicode: Unicode编码
        """
        print(f"[DEBUG] predict 被调用, is_loaded={self.is_loaded}")
        import sys
        sys.stdout.flush()
        
        # 懒加载模型
        self.lazy_load()
        
        print(f"[DEBUG] lazy_load 完成, 开始预处理")
        sys.stdout.flush()

        # 预处理
        input_tensor = self.preprocess(image)
        input_tensor = input_tensor.to(self.device)
        
        print(f"[DEBUG] 预处理完成, 开始推理")
        print(f"[DEBUG] input_tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
        print(f"[DEBUG] input_tensor min: {input_tensor.min().item()}, max: {input_tensor.max().item()}")
        print(f"[DEBUG] model is_loaded: {self.is_loaded}, model type: {type(self.model)}")
        sys.stdout.flush()

        # 测试：先用随机张量测试模型是否可以运行
        print(f"[DEBUG] 测试随机张量推理...")
        sys.stdout.flush()
        test_tensor = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            test_output = self.model(test_tensor)
        print(f"[DEBUG] 随机张量推理成功, output shape: {test_output.shape}")
        sys.stdout.flush()

        # 推理
        with torch.no_grad():
            try:
                print(f"[DEBUG] 进入 torch.no_grad() 上下文")
                sys.stdout.flush()
                output = self.model(input_tensor)
                print(f"[DEBUG] 推理完成, 输出形状: {output.shape}")
                sys.stdout.flush()

                probabilities = torch.softmax(output, dim=1)

                # 获取 top-k 预测
                top_probs, top_indices = torch.topk(probabilities, k=min(top_k, probabilities.size(1)))
                print(f"[DEBUG] Top-k 预测获取成功")
                sys.stdout.flush()

            except Exception as e:
                print(f"推理过程中出错: {e}")
                print(f"错误类型: {type(e)}")
                import traceback
                traceback.print_exc()
                raise

        # 转换为结果列表
        results = []
        for i in range(top_indices.size(1)):
            idx = top_indices[0, i].item()
            prob = top_probs[0, i].item()

            # 获取字符
            character = self.idx_to_char.get(idx, f"unknown_{idx}")

            # 获取 Unicode
            if isinstance(character, str) and len(character) == 1:
                unicode_hex = f"U+{ord(character):04X}"
            else:
                unicode_hex = "N/A"

            results.append({
                "character": character,
                "confidence": float(prob),
                "index": idx,
                "unicode": unicode_hex
            })

        return results

    def get_class_mapping(self) -> Dict[int, str]:
        """获取类别映射"""
        return self.idx_to_char.copy()

    def get_info(self) -> Dict[str, Any]:
        """获取识别器信息"""
        info = super().get_info()
        info.update({
            "num_classes": len(self.idx_to_char),
            "img_size": self.img_size,
            "class_mapping_loaded": len(self.idx_to_char) > 0
        })
        return info