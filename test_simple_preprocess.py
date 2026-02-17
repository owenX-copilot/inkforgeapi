#!/usr/bin/env python3
"""
测试简单的预处理（不改变颜色）
"""

import numpy as np
import cv2
from utils.preprocessing import normalize_handwriting_image

def test_no_color_change():
    """测试不改变颜色的预处理"""
    print("测试：预处理不改变颜色")
    print("=" * 60)

    # 创建测试图像：白底灰字（模拟GUI输出）
    image = np.ones((64, 64), dtype=np.uint8) * 255  # 白色背景
    image[20:44, 20:44] = 60  # 灰色矩形

    print(f"输入图像形状: {image.shape}")
    print(f"输入图像 dtype: {image.dtype}")
    print(f"输入图像值范围: [{np.min(image)}, {np.max(image)}]")
    print(f"输入图像唯一值: {np.unique(image)}")

    # 应用预处理
    result = normalize_handwriting_image(image, enhance_contrast=True)

    print(f"输出图像形状: {result.shape}")
    print(f"输出图像 dtype: {result.dtype}")
    print(f"输出图像值范围: [{np.min(result)}, {np.max(result)}]")
    print(f"输出图像唯一值: {np.unique(result)}")

    # 检查是否改变
    if np.array_equal(image, result):
        print("✓ 测试通过：预处理不改变颜色")
    else:
        print("✗ 测试失败：预处理改变了颜色")
        diff = image - result
        print(f"差异值范围: [{np.min(diff)}, {np.max(diff)}]")

    print()

def test_rgb_to_grayscale():
    """测试RGB到灰度的转换"""
    print("测试：RGB到灰度的转换")
    print("=" * 60)

    # 创建RGB测试图像：白底灰字
    image_rgb = np.ones((64, 64, 3), dtype=np.uint8) * 255  # 白色背景
    image_rgb[20:44, 20:44, :] = 60  # 灰色矩形

    print(f"输入RGB图像形状: {image_rgb.shape}")
    print(f"输入RGB图像 dtype: {image_rgb.dtype}")
    print(f"输入RGB图像值范围: [{np.min(image_rgb)}, {np.max(image_rgb)}]")

    # 应用预处理
    result = normalize_handwriting_image(image_rgb, enhance_contrast=True)

    print(f"输出灰度图像形状: {result.shape}")
    print(f"输出灰度图像 dtype: {result.dtype}")
    print(f"输出灰度图像值范围: [{np.min(result)}, {np.max(result)}]")
    print(f"输出灰度图像唯一值: {np.unique(result)}")

    # 检查转换是否正确
    # RGB白色(255,255,255) -> 灰度255
    # RGB灰色(60,60,60) -> 灰度60
    expected = np.ones((64, 64), dtype=np.uint8) * 255
    expected[20:44, 20:44] = 60

    if np.array_equal(result, expected):
        print("✓ 测试通过：RGB正确转换为灰度")
    else:
        print("✗ 测试失败：RGB到灰度转换不正确")

    print()

def test_float32_conversion():
    """测试float32到uint8的转换"""
    print("测试：float32到uint8的转换")
    print("=" * 60)

    # 创建float32测试图像
    image_float = np.ones((64, 64), dtype=np.float32) * 255.0
    image_float[20:44, 20:44] = 60.0

    print(f"输入float32图像形状: {image_float.shape}")
    print(f"输入float32图像 dtype: {image_float.dtype}")
    print(f"输入float32图像值范围: [{np.min(image_float)}, {np.max(image_float)}]")

    # 应用预处理
    result = normalize_handwriting_image(image_float, enhance_contrast=True)

    print(f"输出uint8图像形状: {result.shape}")
    print(f"输出uint8图像 dtype: {result.dtype}")
    print(f"输出uint8图像值范围: [{np.min(result)}, {np.max(result)}]")
    print(f"输出uint8图像唯一值: {np.unique(result)}")

    # 检查转换是否正确
    expected = np.ones((64, 64), dtype=np.uint8) * 255
    expected[20:44, 20:44] = 60

    if np.array_equal(result, expected):
        print("✓ 测试通过：float32正确转换为uint8")
    else:
        print("✗ 测试失败：float32到uint8转换不正确")

    print()

if __name__ == "__main__":
    test_no_color_change()
    test_rgb_to_grayscale()
    test_float32_conversion()
    print("所有测试完成")