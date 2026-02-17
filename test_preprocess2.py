#!/usr/bin/env python3
"""
测试预处理函数 - 详细版本
"""

import numpy as np
import cv2

def test_resize_issue():
    """测试resize和画布赋值问题"""
    print("测试resize和画布赋值问题")
    print("=" * 60)

    # 创建一个小图像：白底灰字
    small_h, small_w = 10, 10
    small_image = np.ones((small_h, small_w), dtype=np.uint8) * 255  # 白色背景
    small_image[3:7, 3:7] = 60  # 灰色中心区域

    print(f"小图像形状: {small_image.shape}, dtype: {small_image.dtype}")
    print(f"小图像值:")
    print(small_image)
    print()

    # 放大到64x64
    target_h, target_w = 64, 64
    resized = cv2.resize(small_image, (target_w, target_h), interpolation=cv2.INTER_AREA)

    print(f"放大后图像形状: {resized.shape}, dtype: {resized.dtype}")
    print(f"放大后图像值范围: [{np.min(resized)}, {np.max(resized)}]")

    # 检查中心区域的值
    center_h, center_w = target_h // 2, target_w // 2
    print(f"中心区域(30:34, 30:34)的值:")
    print(resized[30:34, 30:34])
    print()

    # 创建float32画布
    canvas = np.ones((target_h, target_w), dtype=np.float32) * 255.0

    print(f"画布形状: {canvas.shape}, dtype: {canvas.dtype}")
    print(f"画布中心区域(30:34, 30:34)赋值前的值:")
    print(canvas[30:34, 30:34])
    print()

    # 将resized放在画布中心
    # 注意：resized已经是64x64，所以直接赋值
    canvas = resized.astype(np.float32)

    print(f"赋值后画布中心区域(30:34, 30:34)的值:")
    print(canvas[30:34, 30:34])
    print(f"赋值后画布值范围: [{np.min(canvas)}, {np.max(canvas)}]")

    # 检查是否有60.0的值
    unique_values = np.unique(canvas)
    print(f"赋值后画布唯一值: {unique_values[:10]}... (共{len(unique_values)}个)")

    # 检查60是否在唯一值中
    if 60.0 in unique_values:
        print("✓ 画布包含60.0（灰色笔迹）")
    else:
        print("✗ 画布不包含60.0！")

    # 检查有多少像素是60
    gray_pixels = np.sum(canvas == 60.0)
    total_pixels = canvas.size
    print(f"灰色像素(60.0)数量: {gray_pixels}/{total_pixels} ({gray_pixels/total_pixels*100:.2f}%)")

    print()

def test_combine_atomic_blocks_simple():
    """简化版combine_atomic_blocks测试"""
    print("简化版combine_atomic_blocks测试")
    print("=" * 60)

    target_size = 64

    # 模拟一个区域：白底灰字
    region_h, region_w = 40, 40
    region = np.ones((region_h, region_w), dtype=np.uint8) * 255
    region[10:30, 10:30] = 60  # 20x20的灰色矩形

    print(f"区域图像形状: {region.shape}, dtype: {region.dtype}")
    print(f"区域图像值范围: [{np.min(region)}, {np.max(region)}]")

    # 转换为灰度（已经是灰度）
    gray = region.copy()

    # 创建画布
    canvas = np.ones((target_size, target_size), dtype=np.float32) * 255

    # 缩放以适应画布
    scale = min(target_size / region_w, target_size / region_h)
    new_w = int(region_w * scale)
    new_h = int(region_h * scale)

    print(f"缩放比例: {scale}")
    print(f"新尺寸: {new_w}x{new_h}")

    # 调整大小
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    print(f"调整后图像形状: {resized.shape}, dtype: {resized.dtype}")
    print(f"调整后图像值范围: [{np.min(resized)}, {np.max(resized)}]")

    # 检查resized中的值
    unique_resized = np.unique(resized)
    print(f"调整后图像唯一值: {unique_resized}")

    # 放在画布中心
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2

    print(f"偏移量: x={x_offset}, y={y_offset}")

    # 直接赋值
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    print(f"最终画布值范围: [{np.min(canvas)}, {np.max(canvas)}]")

    # 检查画布中的值
    unique_canvas = np.unique(canvas)
    print(f"最终画布唯一值: {unique_canvas[:10]}... (共{len(unique_canvas)}个)")

    # 检查是否有60
    has_60 = np.any(canvas == 60.0)
    print(f"画布包含60.0: {has_60}")

    if has_60:
        gray_count = np.sum(canvas == 60.0)
        print(f"灰色像素数量: {gray_count}")
    else:
        # 检查最接近60的值
        diff = np.abs(canvas - 60.0)
        closest_value = canvas[np.unravel_index(np.argmin(diff), canvas.shape)]
        print(f"最接近60的值: {closest_value}")

    print()

if __name__ == "__main__":
    test_resize_issue()
    test_combine_atomic_blocks_simple()