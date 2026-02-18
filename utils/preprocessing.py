"""
图像预处理工具
提供图像解码、预处理和验证功能
"""

import base64
import io
from typing import Tuple, Optional, Union
import numpy as np
from PIL import Image
import cv2


def decode_base64_image(image_data: str) -> np.ndarray:
    """
    解码 base64 图像数据

    Args:
        image_data: base64 编码的图像数据

    Returns:
        numpy 数组表示的图像 (H, W, C) 或 (H, W)

    Raises:
        ValueError: 如果解码失败
    """
    try:
        # 移除可能的 data URL 前缀
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # 解码 base64
        image_bytes = base64.b64decode(image_data)

        # 使用 PIL 读取图像
        image = Image.open(io.BytesIO(image_bytes))

        # 转换为 numpy 数组
        image_array = np.array(image)

        return image_array

    except Exception as e:
        raise ValueError(f"Base64 图像解码失败: {e}")


def normalize_handwriting_image(image: np.ndarray, 
                                stroke_color: int = 81,
                                background_color: int = 255) -> np.ndarray:
    """
    标准化手写图像格式

    将白底任意颜色字转换为白底灰字：
    1. 转换为灰度图（如果是彩色）
    2. 确保uint8类型
    3. 对比度增强：归一化到全范围
    4. 将笔迹转换为指定灰色，背景保持白色

    Args:
        image: 输入图像 (H, W) 或 (H, W, C)
        stroke_color: 笔迹灰度值 (0-255)，默认 81（匹配训练数据）
        background_color: 背景灰度值 (0-255)，默认 255（白色）

    Returns:
        灰度图像 (H, W)，uint8 类型，白底灰字
    """
    # 转换为 numpy 数组
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # 转换为灰度图
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            # RGBA -> 灰度
            # 使用 alpha 通道作为掩码，透明区域变白色背景
            alpha = image[:, :, 3:4]
            rgb = image[:, :, :3]
            # 将透明区域设为白色
            white_bg = np.ones_like(rgb) * 255
            rgb = np.where(alpha > 0, rgb, white_bg)
            gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 3:
            # RGB -> 灰度
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 1:
            gray = image.squeeze()
        else:
            # 其他情况，取第一个通道
            gray = image[:, :, 0]
    else:
        gray = image.copy()

    # 确保是 uint8 类型
    if gray.dtype != np.uint8:
        if np.max(gray) <= 1.0:
            gray = (gray * 255).astype(np.uint8)
        else:
            gray = gray.astype(np.uint8)
    
    # ===== 对比度增强 =====
    # 步骤1: 归一化 - 减去最小值，除以范围，拉满对比度
    min_val = np.min(gray)
    max_val = np.max(gray)
    
    if max_val > min_val:
        # 归一化到 0-255 范围
        gray = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    # 如果 max_val == min_val，说明是纯色图，保持原样
    
    # 步骤2: 使用自适应阈值找到笔迹区域
    # 先用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 使用 Otsu 自适应阈值
    # THRESH_BINARY_INV 因为我们假设背景是亮色（白），笔迹是暗色
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 检查二值化结果是否合理
    # 如果笔迹像素太少（<1%）或太多（>50%），可能阈值不合适
    stroke_ratio = np.sum(binary > 0) / binary.size
    
    if stroke_ratio < 0.005 or stroke_ratio > 0.6:
        # Otsu 结果不理想，使用固定阈值
        # 归一化后，假设背景接近 255，笔迹较暗
        # 使用 240 作为阈值（归一化后的图像）
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # 步骤3: 创建结果图像
    # 使用配置的背景色和笔迹颜色
    result = np.ones_like(gray, dtype=np.uint8) * background_color
    
    # 在笔迹区域（binary > 0）设置为笔迹颜色
    result[binary > 0] = stroke_color

    return result


def validate_image(image: np.ndarray,
                   min_size: Tuple[int, int] = (16, 16),
                   max_size: Tuple[int, int] = (1024, 1024),
                   max_channels: int = 4) -> bool:
    """
    验证图像是否有效

    Args:
        image: 输入图像
        min_size: 最小尺寸 (高度, 宽度)
        max_size: 最大尺寸 (高度, 宽度)
        max_channels: 最大通道数

    Returns:
        图像是否有效
    """
    if not isinstance(image, np.ndarray):
        return False

    if len(image.shape) not in [2, 3]:
        return False

    height, width = image.shape[:2]

    # 检查尺寸
    if height < min_size[0] or width < min_size[1]:
        return False

    if height > max_size[0] or width > max_size[1]:
        return False

    # 检查通道数
    if len(image.shape) == 3:
        channels = image.shape[2]
        if channels > max_channels:
            return False

    # 检查数据类型
    if image.dtype not in [np.uint8, np.float32, np.float64]:
        return False

    return True


def preprocess_image(image: np.ndarray,
                     target_size: Tuple[int, int] = (64, 64),
                     convert_to_grayscale: bool = True,
                     normalize: bool = True,
                     mean: float = 0.5,
                     std: float = 0.5) -> np.ndarray:
    """
    预处理图像

    Args:
        image: 输入图像
        target_size: 目标尺寸 (高度, 宽度)
        convert_to_grayscale: 是否转换为灰度图
        normalize: 是否标准化
        mean: 标准化均值
        std: 标准化标准差

    Returns:
        预处理后的图像

    Raises:
        ValueError: 如果预处理失败
    """
    try:
        # 确保是 numpy 数组
        if not isinstance(image, np.ndarray):
            raise ValueError("输入必须是 numpy 数组")

        # 转换为灰度图
        if convert_to_grayscale:
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                elif image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
                elif image.shape[2] == 1:
                    image = image.squeeze()
            # 确保是二维数组
            if len(image.shape) != 2:
                raise ValueError("灰度转换失败")

        # 调整尺寸
        if image.shape[:2] != target_size:
            image = cv2.resize(image, (target_size[1], target_size[0]),
                              interpolation=cv2.INTER_AREA)

        # 转换为 [0, 1] 范围
        if image.dtype != np.float32:
            image = image.astype(np.float32)
            if np.max(image) > 1.0:  # 假设是 [0, 255] 范围
                image = image / 255.0

        # 标准化
        if normalize:
            image = (image - mean) / std

        return image

    except Exception as e:
        raise ValueError(f"图像预处理失败: {e}")


def auto_crop_character(image: np.ndarray,
                        threshold: float = 0.1,
                        padding: int = 5) -> np.ndarray:
    """
    自动裁剪字符区域

    Args:
        image: 输入图像 (灰度图)
        threshold: 二值化阈值 (0-1)
        padding: 裁剪后添加的边距

    Returns:
        裁剪后的图像
    """
    if len(image.shape) != 2:
        raise ValueError("输入必须是灰度图像")

    # 二值化
    if np.max(image) <= 1.0:
        binary = (image > threshold).astype(np.uint8) * 255
    else:
        binary = (image > threshold * 255).astype(np.uint8) * 255

    # 找到非零像素的位置
    non_zero = np.where(binary > 0)
    if len(non_zero[0]) == 0 or len(non_zero[1]) == 0:
        return image  # 没有找到字符，返回原图

    # 计算边界框
    y_min, y_max = np.min(non_zero[0]), np.max(non_zero[0])
    x_min, x_max = np.min(non_zero[1]), np.max(non_zero[1])

    # 添加边距
    y_min = max(0, y_min - padding)
    y_max = min(image.shape[0] - 1, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(image.shape[1] - 1, x_max + padding)

    # 裁剪
    cropped = image[y_min:y_max+1, x_min:x_max+1]

    return cropped