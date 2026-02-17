"""
原子切割模块 (Atomic Segmentation)
基于 X-Projection Clustering 实现手写行图像的原子块切割

核心理念：不要尝试直接切出"字"，而是切出"不可再分的最小笔画组"（原子块）
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AtomicBlock:
    """原子块数据结构"""
    image: np.ndarray  # 原子块图像
    x_start: int       # 在原图中的X起始位置
    x_end: int         # 在原图中的X结束位置
    y_start: int       # 在原图中的Y起始位置
    y_end: int         # 在原图中的Y结束位置
    width: int         # 宽度
    height: int        # 高度
    
    def __post_init__(self):
        self.width = self.x_end - self.x_start
        self.height = self.y_end - self.y_start


def compute_x_projection(binary_image: np.ndarray) -> np.ndarray:
    """
    计算图像的X轴投影（垂直投影）
    统计每一列的像素数量
    
    Args:
        binary_image: 二值化图像 (H, W)，前景为255，背景为0
        
    Returns:
        X轴投影数组，长度为W
    """
    return np.sum(binary_image > 0, axis=0)


def find_atomic_blocks_by_projection(
    binary_image: np.ndarray,
    min_gap_threshold: int = 3,
    min_block_width: int = 2
) -> List[Tuple[int, int]]:
    """
    通过X轴投影找到原子块的X范围
    
    Args:
        binary_image: 二值化图像
        min_gap_threshold: 最小间隙阈值，小于此值的间隙不认为是断开
        min_block_width: 最小原子块宽度
        
    Returns:
        原子块X范围列表 [(x_start, x_end), ...]
    """
    projection = compute_x_projection(binary_image)
    
    # 找到有像素的列
    has_pixel = projection > 0
    
    blocks = []
    in_block = False
    block_start = 0
    gap_count = 0
    
    for i, pixel_present in enumerate(has_pixel):
        if pixel_present:
            if not in_block:
                # 开始新块
                in_block = True
                block_start = i
            gap_count = 0
        else:
            if in_block:
                gap_count += 1
                if gap_count > min_gap_threshold:
                    # 间隙足够大，结束当前块
                    block_end = i - gap_count
                    if block_end - block_start >= min_block_width:
                        blocks.append((block_start, block_end))
                    in_block = False
                    gap_count = 0
    
    # 处理最后一个块
    if in_block:
        block_end = len(projection) - gap_count
        if block_end - block_start >= min_block_width:
            blocks.append((block_start, block_end))
    
    return blocks


def oversplit_wide_block(
    binary_image: np.ndarray,
    x_start: int,
    x_end: int,
    aspect_ratio_threshold: float = 1.2
) -> List[Tuple[int, int]]:
    """
    对过宽的原子块进行过分割
    如果宽度超过高度的1.2倍，在垂直投影的波谷处强行劈开
    
    Args:
        binary_image: 二值化图像
        x_start: 块的X起始位置
        x_end: 块的X结束位置
        aspect_ratio_threshold: 宽高比阈值
        
    Returns:
        分割后的X范围列表
    """
    # 提取块区域
    block_region = binary_image[:, x_start:x_end]
    
    # 计算块的高度
    y_projection = np.sum(block_region > 0, axis=1)
    y_nonzero = np.where(y_projection > 0)[0]
    
    if len(y_nonzero) == 0:
        return [(x_start, x_end)]
    
    block_height = y_nonzero[-1] - y_nonzero[0] + 1
    block_width = x_end - x_start
    
    # 如果宽高比不超过阈值，不需要分割
    if block_width <= block_height * aspect_ratio_threshold:
        return [(x_start, x_end)]
    
    # 在块内寻找垂直投影的波谷
    x_projection = compute_x_projection(block_region)
    
    if len(x_projection) < 5:
        return [(x_start, x_end)]
    
    # 平滑投影
    kernel_size = max(3, len(x_projection) // 10)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = np.convolve(x_projection, np.ones(kernel_size)/kernel_size, mode='same')
    
    # 寻找局部最小值（波谷）
    valleys = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] < smoothed[i-1] and smoothed[i] < smoothed[i+1]:
            # 波谷必须有一定深度
            if smoothed[i] < np.max(smoothed) * 0.5:
                valleys.append(i)
    
    if not valleys:
        return [(x_start, x_end)]
    
    # 选择最深的波谷进行分割
    deepest_valley = min(valleys, key=lambda v: smoothed[v])
    split_point = x_start + deepest_valley
    
    return [(x_start, split_point), (split_point, x_end)]


def extract_atomic_block(
    image: np.ndarray,
    x_start: int,
    x_end: int,
    padding: int = 2
) -> AtomicBlock:
    """
    从原图中提取原子块
    
    Args:
        image: 原始图像
        x_start: X起始位置
        x_end: X结束位置
        padding: 边距
        
    Returns:
        AtomicBlock 对象
    """
    height, width = image.shape[:2]
    
    # 提取X范围
    x_start_pad = max(0, x_start - padding)
    x_end_pad = min(width, x_end + padding)
    
    block_region = image[:, x_start_pad:x_end_pad]
    
    # 计算Y范围（找到有像素的区域）
    if len(block_region.shape) == 3:
        gray = cv2.cvtColor(block_region, cv2.COLOR_RGB2GRAY)
    else:
        gray = block_region
    
    # 二值化 - 注意：对于手写图像，背景是白色(255)，笔迹是深色(接近0)
    # 所以需要反转来正确检测笔迹
    if np.max(gray) > 1:
        # 使用自适应阈值或固定阈值
        # 对于白底黑字，使用 THRESH_BINARY_INV
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        gray_uint8 = (gray * 255).astype(np.uint8)
        _, binary = cv2.threshold(gray_uint8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    y_projection = np.sum(binary > 0, axis=1)
    y_nonzero = np.where(y_projection > 0)[0]
    
    if len(y_nonzero) == 0:
        # 没有找到像素，返回整个区域
        y_start_abs = 0
        y_end_abs = height
    else:
        y_start = y_nonzero[0]
        y_end = y_nonzero[-1] + 1
        y_start_abs = max(0, y_start - padding)
        y_end_abs = min(height, y_end + padding)
    
    # 提取最终区域
    final_block = image[y_start_abs:y_end_abs, x_start_pad:x_end_pad]
    
    return AtomicBlock(
        image=final_block,
        x_start=int(x_start_pad),
        x_end=int(x_end_pad),
        y_start=int(y_start_abs),
        y_end=int(y_end_abs),
        width=int(x_end_pad - x_start_pad),
        height=int(y_end_abs - y_start_abs)
    )


def segment_image_to_atomic_blocks(
    image: np.ndarray,
    min_gap_threshold: Optional[int] = None,
    aspect_ratio_threshold: float = 1.2,
    padding: int = 2
) -> List[AtomicBlock]:
    """
    将手写行图像分割为原子块序列
    
    Args:
        image: 输入图像（灰度或彩色）
        min_gap_threshold: 最小间隙阈值，默认为图像高度的5%
        aspect_ratio_threshold: 过分割的宽高比阈值
        padding: 提取时的边距
        
    Returns:
        原子块列表，从左到右排列
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # 确保是uint8
    if gray.dtype != np.uint8:
        if np.max(gray) <= 1:
            gray = (gray * 255).astype(np.uint8)
        else:
            gray = gray.astype(np.uint8)
    
    # 二值化 - 对于白底黑字的手写图像，使用 THRESH_BINARY_INV
    # 这样笔迹（深色）变成白色（255），背景变成黑色（0）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 设置默认间隙阈值
    height = binary.shape[0]
    if min_gap_threshold is None:
        min_gap_threshold = max(2, int(height * 0.05))
    
    # 第一步：通过X投影找到初始原子块
    initial_blocks = find_atomic_blocks_by_projection(
        binary, 
        min_gap_threshold=min_gap_threshold,
        min_block_width=2
    )
    
    # 第二步：对过宽的块进行过分割
    final_block_ranges = []
    for x_start, x_end in initial_blocks:
        sub_blocks = oversplit_wide_block(
            binary, x_start, x_end, 
            aspect_ratio_threshold=aspect_ratio_threshold
        )
        final_block_ranges.extend(sub_blocks)
    
    # 第三步：提取原子块图像
    atomic_blocks = []
    for x_start, x_end in final_block_ranges:
        block = extract_atomic_block(image, x_start, x_end, padding)
        atomic_blocks.append(block)
    
    return atomic_blocks


def visualize_atomic_blocks(
    image: np.ndarray,
    blocks: List[AtomicBlock],
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    可视化原子块分割结果
    
    Args:
        image: 原始图像
        blocks: 原子块列表
        output_path: 输出路径（可选）
        
    Returns:
        可视化图像
    """
    # 转换为彩色图像
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        vis_image = image.copy()
    
    # 为每个块绘制边界框
    colors = [
        (255, 0, 0),    # 红
        (0, 255, 0),    # 绿
        (0, 0, 255),    # 蓝
        (255, 255, 0),  # 青
        (255, 0, 255),  # 品红
        (0, 255, 255),  # 黄
    ]
    
    for i, block in enumerate(blocks):
        color = colors[i % len(colors)]
        cv2.rectangle(
            vis_image,
            (block.x_start, block.y_start),
            (block.x_end, block.y_end),
            color,
            2
        )
        # 标注块编号
        cv2.putText(
            vis_image,
            str(i),
            (block.x_start, block.y_start - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
    
    if output_path:
        cv2.imwrite(output_path, vis_image)
    
    return vis_image


if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
        image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is not None:
            blocks = segment_image_to_atomic_blocks(image)
            print(f"检测到 {len(blocks)} 个原子块:")
            for i, block in enumerate(blocks):
                print(f"  块 {i}: x=[{block.x_start}, {block.x_end}], "
                      f"y=[{block.y_start}, {block.y_end}], "
                      f"尺寸={block.width}x{block.height}")
            
            # 可视化
            vis = visualize_atomic_blocks(image, blocks, "atomic_blocks_vis.png")
            print("可视化结果已保存到 atomic_blocks_vis.png")
        else:
            print(f"无法读取图像: {test_image_path}")
    else:
        print("用法: python atomic_segmentation.py <图像路径>")
