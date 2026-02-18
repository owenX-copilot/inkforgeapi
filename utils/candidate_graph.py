"""
Candidate Graph Builder - Candidate Graph Construction
Builds a DAG of candidate characters from atomic blocks.

Core concept: A Chinese character may consist of 1, 2, or 3 consecutive atomic blocks.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

from .atomic_segmentation import AtomicBlock


@dataclass
class CandidateEdge:
    """Candidate edge representing a possible character"""
    start_block: int       # Starting block index (inclusive)
    end_block: int         # Ending block index (exclusive)
    image: np.ndarray      # Combined image of the blocks
    width: int             # Width of the combined region
    height: int            # Height of the combined region
    x_start: int           # X position in original image
    x_end: int             # X end position in original image
    y_start: int = 0       # Y position in original image
    y_end: int = 0         # Y end position in original image
    cost: float = float('inf')  # Recognition cost (to be filled later)
    predictions: List[Dict] = field(default_factory=list)  # Recognition results
    gap_penalty: float = 0.0   # Penalty for gaps between blocks
    
    @property
    def num_blocks(self) -> int:
        return self.end_block - self.start_block


@dataclass
class CandidateGraph:
    """
    Candidate Graph (DAG)
    Nodes: positions between blocks (0, 1, 2, ..., n)
    Edges: candidate characters spanning one or more blocks
    """
    num_blocks: int                              # Number of atomic blocks
    edges: List[CandidateEdge] = field(default_factory=list)  # All candidate edges
    adjacency: Dict[int, List[int]] = field(default_factory=dict)  # node -> list of edge indices
    
    def add_edge(self, edge: CandidateEdge):
        """Add an edge to the graph"""
        edge_idx = len(self.edges)
        self.edges.append(edge)
        
        if edge.start_block not in self.adjacency:
            self.adjacency[edge.start_block] = []
        self.adjacency[edge.start_block].append(edge_idx)
    
    def get_edges_from(self, node: int) -> List[CandidateEdge]:
        """Get all edges starting from a node"""
        edge_indices = self.adjacency.get(node, [])
        return [self.edges[i] for i in edge_indices]


def combine_atomic_blocks(
    blocks: List[AtomicBlock],
    start_idx: int,
    end_idx: int,
    target_height: int = 64,
    target_width: int = 64,
    original_image: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    """
    Combine multiple atomic blocks into a single image
    
    Args:
        blocks: List of atomic blocks
        start_idx: Starting block index (inclusive)
        end_idx: Ending block index (exclusive)
        target_height: Target height for the combined image
        target_width: Target width for the combined image
        original_image: Original full image (optional, for better extraction)
        
    Returns:
        Combined and resized image, or None if invalid
    """
    if start_idx < 0 or end_idx > len(blocks) or start_idx >= end_idx:
        return None
    
    # Get the bounding box of all blocks
    x_start = blocks[start_idx].x_start
    x_end = blocks[end_idx - 1].x_end
    
    # Find the min y_start and max y_end across all blocks
    y_start = min(blocks[i].y_start for i in range(start_idx, end_idx))
    y_end = max(blocks[i].y_end for i in range(start_idx, end_idx))
    
    # Extract from original image if available
    if original_image is not None:
        height, width = original_image.shape[:2]

        # 添加padding留白，避免字贴边
        padding = 10  # 10像素的padding
        padded_y_start = max(0, y_start - padding)
        padded_y_end = min(height, y_end + padding)
        padded_x_start = max(0, x_start - padding)
        padded_x_end = min(width, x_end + padding)

        region = original_image[
            padded_y_start:padded_y_end,
            padded_x_start:padded_x_end
        ]
    else:
        # Combine block images horizontally
        block_images = [blocks[i].image for i in range(start_idx, end_idx)]
        
        # Find max height
        max_h = max(img.shape[0] for img in block_images)
        
        # Pad each block to same height and concatenate
        padded = []
        for img in block_images:
            if len(img.shape) == 3:
                h, w, c = img.shape
                pad = np.zeros((max_h, w, c), dtype=img.dtype)
            else:
                h, w = img.shape
                pad = np.zeros((max_h, w), dtype=img.dtype)
            
            # Center vertically
            y_offset = (max_h - h) // 2
            pad[y_offset:y_offset+h, :] = img
            padded.append(pad)
        
        region = np.hstack(padded)
    
    if region.size == 0:
        return None
    
    # Convert to grayscale if needed
    if len(region.shape) == 3:
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    else:
        gray = region.copy()
    
    # 直接拉伸到 target_size，不保持宽高比
    # 这样可以让 "一" 这种非常扁的字变厚，更容易识别
    resized = cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # 返回画布，不改变颜色
    # 前端已经确保是白底灰字（背景255，笔迹60）
    return resized.astype(np.float32)


def geometric_filter(
    combined_width: int,
    combined_height: int,
    num_blocks: int,
    max_width_height_ratio: float = 2.0,
    min_width_height_ratio: float = 0.3
) -> bool:
    """
    Geometric filter to eliminate unreasonable combinations
    
    Args:
        combined_width: Width of the combined region
        combined_height: Height of the combined region
        num_blocks: Number of blocks being combined
        max_width_height_ratio: Maximum allowed width/height ratio
        min_width_height_ratio: Minimum allowed width/height ratio
        
    Returns:
        True if the combination passes the filter, False otherwise
    """
    if combined_height <= 0:
        return False
    
    ratio = combined_width / combined_height
    
    # Single block: allow very wide range (could be "一", "二"的一横, "1", "l", etc.)
    # 手写汉字中，"一"字的宽高比可能达到 5-10
    if num_blocks == 1:
        return 0.15 <= ratio <= 6.0
    
    # Two blocks: moderate constraint
    if num_blocks == 2:
        return 0.3 <= ratio <= 3.0
    
    # Three blocks: stricter constraint
    if num_blocks == 3:
        return 0.4 <= ratio <= 2.0
    
    # More than 3 blocks: very strict
    return 0.5 <= ratio <= 1.5


def calculate_gap_penalty(
    blocks: List[AtomicBlock],
    start_idx: int,
    end_idx: int
) -> float:
    """
    Calculate gap penalty for combining multiple blocks
    
    Args:
        blocks: List of atomic blocks
        start_idx: Starting block index
        end_idx: Ending block index (exclusive)
        
    Returns:
        Gap penalty (0 for single block, higher for larger gaps)
    """
    num_blocks = end_idx - start_idx
    if num_blocks <= 1:
        return 0.0
    
    # Base merge penalty: combining blocks itself has cost
    # This prevents over-merging of separate characters
    merge_penalty = 1.5 * (num_blocks - 1)  # 1.5 per additional block
    
    total_gap = 0
    for i in range(start_idx, end_idx - 1):
        gap = blocks[i + 1].x_start - blocks[i].x_end
        total_gap += max(0, gap)
    
    # Normalize by average block width
    avg_width = sum(blocks[i].width for i in range(start_idx, end_idx)) / num_blocks
    if avg_width > 0:
        normalized_gap = total_gap / avg_width
    else:
        normalized_gap = 0
    
    # Penalty increases with gap size
    # 0 gap = 0 penalty, 1x width gap = 2.0 penalty
    gap_penalty = normalized_gap * 2.0
    
    return merge_penalty + gap_penalty


def build_candidate_graph(
    blocks: List[AtomicBlock],
    original_image: Optional[np.ndarray] = None,
    max_blocks_per_char: int = 3,
    target_size: int = 64,
    enable_geometric_filter: bool = True
) -> CandidateGraph:
    """
    Build a candidate graph from atomic blocks
    
    Args:
        blocks: List of atomic blocks (ordered left to right)
        original_image: Original full image (for better extraction)
        max_blocks_per_char: Maximum number of blocks that can form one character
        target_size: Target size for character images
        enable_geometric_filter: Whether to apply geometric filtering
        
    Returns:
        CandidateGraph object
    """
    n = len(blocks)
    graph = CandidateGraph(num_blocks=n)
    
    if n == 0:
        return graph
    
    print(f"[DEBUG GRAPH] 构建候选图, 块数={n}")
    for i, b in enumerate(blocks):
        print(f"[DEBUG GRAPH]   块 {i}: x=[{b.x_start},{b.x_end}], y=[{b.y_start},{b.y_end}], 尺寸={b.width}x{b.height}")
    
    # Generate all possible combinations
    for start in range(n):
        for span in range(1, min(max_blocks_per_char, n - start) + 1):
            end = start + span
            
            # Get combined dimensions
            x_start = blocks[start].x_start
            x_end = blocks[end - 1].x_end
            
            # Calculate combined height (max across blocks)
            y_start = min(blocks[i].y_start for i in range(start, end))
            y_end = max(blocks[i].y_end for i in range(start, end))
            combined_height = y_end - y_start
            combined_width = x_end - x_start
            
            print(f"[DEBUG GRAPH] 尝试组合: 块[{start}->{end}], 尺寸={combined_width}x{combined_height}")
            
            # Geometric filter
            if enable_geometric_filter:
                if not geometric_filter(combined_width, combined_height, span):
                    ratio = combined_width / combined_height if combined_height > 0 else 0
                    print(f"[DEBUG GRAPH]   被几何过滤器拒绝: 宽高比={ratio:.2f}")
                    continue
                else:
                    print(f"[DEBUG GRAPH]   通过几何过滤器")
            
            # Combine blocks
            combined_image = combine_atomic_blocks(
                blocks, start, end,
                target_height=target_size,
                target_width=target_size,
                original_image=original_image
            )
            
            if combined_image is None:
                print(f"[DEBUG GRAPH]   图像组合失败")
                continue
            
            # Calculate gap penalty for multi-block combinations
            gap_penalty = calculate_gap_penalty(blocks, start, end)
            
            # Create candidate edge - convert to Python int for JSON serialization
            # 注意：这里的x_start, x_end, y_start, y_end是添加padding前的原始坐标
            # 但实际提取的图像区域已经包含了padding
            edge = CandidateEdge(
                start_block=start,
                end_block=end,
                image=combined_image,
                width=int(combined_width),
                height=int(combined_height),
                x_start=int(x_start),
                x_end=int(x_end),
                y_start=int(y_start),
                y_end=int(y_end),
                gap_penalty=gap_penalty
            )
            
            graph.add_edge(edge)
            print(f"[DEBUG GRAPH]   添加边: {start}->{end}, 间隙惩罚={gap_penalty:.2f}")
    
    return graph


def prepare_batch_for_inference(
    graph: CandidateGraph
) -> Tuple[np.ndarray, List[int]]:
    """
    Prepare all candidate images as a batch for model inference
    
    Args:
        graph: Candidate graph
        
    Returns:
        Tuple of (batch_array, edge_indices)
        batch_array: shape (N, 1, 64, 64) for N candidates
        edge_indices: list of edge indices corresponding to each batch item
    """
    images = []
    edge_indices = []
    
    for idx, edge in enumerate(graph.edges):
        # Normalize image to [0, 1]
        img = edge.image.astype(np.float32)
        if np.max(img) > 1:
            img = img / 255.0
        
        # Add channel dimension
        img = img[np.newaxis, :, :]  # (1, 64, 64)
        
        images.append(img)
        edge_indices.append(idx)
    
    if not images:
        return np.array([]).reshape(0, 1, 64, 64), []
    
    batch = np.stack(images, axis=0)  # (N, 1, 64, 64)
    return batch, edge_indices


def visualize_candidate_graph(
    graph: CandidateGraph,
    blocks: List[AtomicBlock],
    original_image: np.ndarray,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize the candidate graph
    
    Args:
        graph: Candidate graph
        blocks: Atomic blocks
        original_image: Original image
        output_path: Output path (optional)
        
    Returns:
        Visualization image
    """
    # Convert to color
    if len(original_image.shape) == 2:
        vis = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        vis = original_image.copy()
    
    # Draw each edge with different color based on number of blocks
    colors = {
        1: (0, 255, 0),    # Green for single block
        2: (255, 165, 0),  # Orange for two blocks
        3: (255, 0, 255),  # Magenta for three blocks
    }
    
    for edge in graph.edges:
        num_blocks = edge.num_blocks
        color = colors.get(num_blocks, (128, 128, 128))
        
        # Draw rectangle
        cv2.rectangle(
            vis,
            (edge.x_start, 0),
            (edge.x_end, vis.shape[0] - 1),
            color,
            1
        )
    
    if output_path:
        cv2.imwrite(output_path, vis)
    
    return vis


if __name__ == "__main__":
    # Test code
    from atomic_segmentation import segment_image_to_atomic_blocks, AtomicBlock
    import sys
    
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
        image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is not None:
            # Segment
            blocks = segment_image_to_atomic_blocks(image)
            print(f"Found {len(blocks)} atomic blocks")
            
            # Build graph
            graph = build_candidate_graph(blocks, image)
            print(f"Built graph with {len(graph.edges)} candidate edges")
            
            for i, edge in enumerate(graph.edges):
                print(f"  Edge {i}: blocks [{edge.start_block}, {edge.end_block}), "
                      f"size={edge.width}x{edge.height}")
            
            # Visualize
            vis = visualize_candidate_graph(graph, blocks, image, "candidate_graph_vis.png")
            print("Visualization saved to candidate_graph_vis.png")
        else:
            print(f"Cannot read image: {test_image_path}")
    else:
        print("Usage: python candidate_graph.py <image_path>")