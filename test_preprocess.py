#!/usr/bin/env python3
"""
测试预处理函数
"""

import numpy as np
import cv2
from utils.preprocessing import normalize_handwriting_image

def test_white_background_gray_stroke():
    """测试白底灰字输入"""
    print("测试1: 白底灰字输入")

    # 创建测试图像：64x64，白底(255)，中间有灰色(60)的矩形
    image = np.ones((64, 64), dtype=np.uint8) * 255  # 白色背景
    image[20:44, 20:44] = 60  # 灰色矩形

    print(f"输入图像形状: {image.shape}")
    print(f"输入图像 dtype: {image.dtype}")
    print(f"输入图像值范围: [{np.min(image)}, {np.max(image)}]")
    print(f"输入图像均值: {np.mean(image):.2f}")

    # 应用预处理
    result = normalize_handwriting_image(image, enhance_contrast=True)

    print(f"输出图像形状: {result.shape}")
    print(f"输出图像 dtype: {result.dtype}")
    print(f"输出图像值范围: [{np.min(result)}, {np.max(result)}]")
    print(f"输出图像均值: {np.mean(result):.2f}")

    # 检查结果
    unique_values = np.unique(result)
    print(f"输出图像唯一值: {unique_values}")

    # 应该只有两个值：255（白色背景）和60（灰色笔迹）
    if len(unique_values) == 2 and 255 in unique_values and 60 in unique_values:
        print("✓ 测试通过：输出为白底灰字")
    else:
        print("✗ 测试失败：输出不符合预期")

    print()

def test_white_background_black_stroke():
    """测试白底黑字输入"""
    print("测试2: 白底黑字输入")

    # 创建测试图像：64x64，白底(255)，中间有黑色(0)的矩形
    image = np.ones((64, 64), dtype=np.uint8) * 255  # 白色背景
    image[20:44, 20:44] = 0  # 黑色矩形

    print(f"输入图像形状: {image.shape}")
    print(f"输入图像 dtype: {image.dtype}")
    print(f"输入图像值范围: [{np.min(image)}, {np.max(image)}]")
    print(f"输入图像均值: {np.mean(image):.2f}")

    # 应用预处理
    result = normalize_handwriting_image(image, enhance_contrast=True)

    print(f"输出图像形状: {result.shape}")
    print(f"输出图像 dtype: {result.dtype}")
    print(f"输出图像值范围: [{np.min(result)}, {np.max(result)}]")
    print(f"输出图像均值: {np.mean(result):.2f}")

    # 检查结果
    unique_values = np.unique(result)
    print(f"输出图像唯一值: {unique_values}")

    # 应该只有两个值：255（白色背景）和60（灰色笔迹）
    if len(unique_values) == 2 and 255 in unique_values and 60 in unique_values:
        print("✓ 测试通过：输出为白底灰字")
    else:
        print("✗ 测试失败：输出不符合预期")

    print()

def test_white_background_light_gray_stroke():
    """测试白底浅灰字输入"""
    print("测试3: 白底浅灰字输入（笔迹颜色接近背景）")

    # 创建测试图像：64x64，白底(255)，中间有浅灰色(200)的矩形
    image = np.ones((64, 64), dtype=np.uint8) * 255  # 白色背景
    image[20:44, 20:44] = 200  # 浅灰色矩形

    print(f"输入图像形状: {image.shape}")
    print(f"输入图像 dtype: {image.dtype}")
    print(f"输入图像值范围: [{np.min(image)}, {np.max(image)}]")
    print(f"输入图像均值: {np.mean(image):.2f}")

    # 应用预处理
    result = normalize_handwriting_image(image, enhance_contrast=True)

    print(f"输出图像形状: {result.shape}")
    print(f"输出图像 dtype: {result.dtype}")
    print(f"输出图像值范围: [{np.min(result)}, {np.max(result)}]")
    print(f"输出图像均值: {np.mean(result):.2f}")

    # 检查结果
    unique_values = np.unique(result)
    print(f"输出图像唯一值: {unique_values}")

    print()

def test_combine_atomic_blocks_logic():
    """测试combine_atomic_blocks函数中的画布逻辑"""
    print("测试4: 测试combine_atomic_blocks中的画布逻辑")

    # 模拟combine_atomic_blocks函数中的逻辑
    target_height = 64
    target_width = 64

    # 创建白色画布
    canvas = np.ones((target_height, target_width), dtype=np.float32) * 255
    print(f"画布形状: {canvas.shape}, dtype: {canvas.dtype}")
    print(f"画布值范围: [{np.min(canvas)}, {np.max(canvas)}]")

    # 模拟一个区域图像（白底灰字）
    region_height = 40
    region_width = 40
    region = np.ones((region_height, region_width), dtype=np.uint8) * 255  # 白色背景
    region[10:30, 10:30] = 60  # 灰色矩形

    print(f"区域图像形状: {region.shape}, dtype: {region.dtype}")
    print(f"区域图像值范围: [{np.min(region)}, {np.max(region)}]")

    # 调整大小
    scale = min(target_width / region_width, target_height / region_height)
    new_w = int(region_width * scale)
    new_h = int(region_height * scale)

    resized = cv2.resize(region, (new_w, new_h), interpolation=cv2.INTER_AREA)
    print(f"调整后图像形状: {resized.shape}, dtype: {resized.dtype}")
    print(f"调整后图像值范围: [{np.min(resized)}, {np.max(resized)}]")

    # 放在画布上
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    print(f"最终画布值范围: [{np.min(canvas)}, {np.max(canvas)}]")
    print(f"最终画布在笔迹区域的值: {canvas[y_offset+15, x_offset+15]} (应该是60.0)")
    print(f"最终画布在背景区域的值: {canvas[0, 0]} (应该是255.0)")

    print()

if __name__ == "__main__":
    print("=" * 60)
    print("预处理函数测试")
    print("=" * 60)
    print()

    test_white_background_gray_stroke()
    test_white_background_black_stroke()
    test_white_background_light_gray_stroke()
    test_combine_atomic_blocks_logic()

    print("测试完成")