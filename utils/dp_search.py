"""
DP Path Search Module - Dynamic Programming for Optimal Segmentation
Finds the globally optimal segmentation path through the candidate graph.

Core concept: Use DP to find the path from start to end with minimum total cost.
DP transition: dp[i] = min(dp[j] + cost(j, i)) where j is a previous breakpoint.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import heapq

from .candidate_graph import CandidateGraph, CandidateEdge


@dataclass
class PathNode:
    """Node in the optimal path"""
    edge: CandidateEdge          # The edge taken
    cumulative_cost: float       # Cumulative cost up to this point
    character: str               # Recognized character
    confidence: float            # Recognition confidence


@dataclass
class RecognitionResult:
    """Final recognition result"""
    text: str                    # Recognized text
    characters: List[Dict]       # List of character details
    total_cost: float            # Total path cost
    num_blocks: int              # Number of atomic blocks
    num_characters: int          # Number of recognized characters


def compute_edge_cost(
    predictions: List[Dict],
    shape_penalty_weight: float = 0.1,
    edge: Optional[CandidateEdge] = None
) -> float:
    """
    Compute the cost of an edge based on recognition results
    
    Cost = -log(confidence) + shape_penalty + gap_penalty
    
    Args:
        predictions: List of prediction dictionaries with 'confidence' key
        shape_penalty_weight: Weight for shape-based penalty
        edge: The candidate edge (for shape penalty and gap penalty calculation)
        
    Returns:
        Cost value (lower is better)
    """
    if not predictions:
        return 10.0  # High cost for no predictions
    
    # Get best confidence
    best_confidence = predictions[0].get('confidence', 0.01)
    
    # Clamp confidence to avoid log(0)
    confidence = max(best_confidence, 0.001)
    
    # Base cost: negative log probability
    base_cost = -np.log(confidence)
    
    # Shape penalty: penalize unusual aspect ratios
    shape_penalty = 0.0
    if edge is not None and edge.height > 0:
        ratio = edge.width / edge.height
        # Optimal ratio for Chinese characters is around 1.0
        # Penalize deviations
        shape_penalty = shape_penalty_weight * abs(ratio - 1.0)
    
    # Gap penalty: penalize combining blocks with large gaps
    gap_penalty = 0.0
    if edge is not None:
        gap_penalty = edge.gap_penalty
    
    total_cost = base_cost + shape_penalty + gap_penalty
    print(f"[DEBUG COST] 边 {edge.start_block if edge else '?'}->{edge.end_block if edge else '?'}: "
          f"基础={base_cost:.3f}, 形状={shape_penalty:.3f}, 间隙={gap_penalty:.3f}, 总计={total_cost:.3f}")
    
    return total_cost


def dp_search(
    graph: CandidateGraph,
    shape_penalty_weight: float = 0.1
) -> Tuple[List[PathNode], float]:
    """
    Dynamic programming search for optimal path
    
    Args:
        graph: Candidate graph with costs filled in
        shape_penalty_weight: Weight for shape penalty
        
    Returns:
        Tuple of (optimal_path, total_cost)
        optimal_path: List of PathNode from start to end
        total_cost: Total cost of the path
    """
    n = graph.num_blocks
    
    print(f"[DEBUG DP] 开始DP搜索, 块数={n}, 边数={len(graph.edges)}")
    
    if n == 0:
        return [], 0.0
    
    # Initialize DP arrays
    # dp[i] = minimum cost to reach position i
    dp = [float('inf')] * (n + 1)
    dp[0] = 0.0
    
    # parent[i] = (prev_position, edge_index) for backtracking
    parent = [None] * (n + 1)
    
    # Compute costs for all edges if not already computed
    for edge in graph.edges:
        if edge.cost == float('inf'):
            edge.cost = compute_edge_cost(
                edge.predictions,
                shape_penalty_weight=shape_penalty_weight,
                edge=edge
            )
        print(f"[DEBUG DP] 边 {edge.start_block}->{edge.end_block}: 代价={edge.cost:.3f}")
    
    # DP forward pass
    for pos in range(n):
        if dp[pos] == float('inf'):
            print(f"[DEBUG DP] 位置 {pos} 不可达")
            continue
        
        edges_from_pos = graph.get_edges_from(pos)
        print(f"[DEBUG DP] 位置 {pos} 可达, 从此处出发的边数={len(edges_from_pos)}")
        
        # Try all edges starting from this position
        for edge in edges_from_pos:
            new_pos = edge.end_block
            new_cost = dp[pos] + edge.cost
            
            if new_cost < dp[new_pos]:
                dp[new_pos] = new_cost
                parent[new_pos] = (pos, edge)
                print(f"[DEBUG DP]   更新位置 {new_pos}: 代价={new_cost:.3f}")
    
    print(f"[DEBUG DP] DP数组: {[f'{c:.2f}' if c != float('inf') else 'inf' for c in dp]}")
    
    # Check if we reached the end
    if dp[n] == float('inf'):
        print(f"[DEBUG DP] 无法到达终点 {n}")
        # No valid path found - try to find partial path
        # Find the furthest reachable position
        max_reachable = 0
        for i in range(n, -1, -1):
            if dp[i] < float('inf'):
                max_reachable = i
                break
        
        if max_reachable == 0:
            # Cannot reach any position, return empty
            return [], float('inf')
        
        # Use partial path
        n = max_reachable
    
    # Backtrack to find optimal path
    path = []
    pos = n
    
    while pos > 0 and parent[pos] is not None:
        prev_pos, edge = parent[pos]
        
        # Get best character from predictions
        if edge.predictions:
            best_pred = edge.predictions[0]
            character = best_pred.get('character', '?')
            confidence = best_pred.get('confidence', 0.0)
        else:
            character = '?'
            confidence = 0.0
        
        path_node = PathNode(
            edge=edge,
            cumulative_cost=dp[pos],
            character=character,
            confidence=confidence
        )
        
        path.append(path_node)
        pos = prev_pos
    
    # Reverse to get left-to-right order
    path.reverse()
    
    return path, dp[n]


def dijkstra_search(
    graph: CandidateGraph,
    shape_penalty_weight: float = 0.1
) -> Tuple[List[PathNode], float]:
    """
    Dijkstra's algorithm for optimal path search
    Alternative to DP, useful for sparse graphs
    
    Args:
        graph: Candidate graph
        shape_penalty_weight: Weight for shape penalty
        
    Returns:
        Tuple of (optimal_path, total_cost)
    """
    n = graph.num_blocks
    
    if n == 0:
        return [], 0.0
    
    # Compute costs for all edges
    for edge in graph.edges:
        if edge.cost == float('inf'):
            edge.cost = compute_edge_cost(
                edge.predictions,
                shape_penalty_weight=shape_penalty_weight,
                edge=edge
            )
    
    # Priority queue: (cost, position, path)
    # Use heap for efficient minimum extraction
    pq = [(0.0, 0, [])]
    
    # Visited positions with their minimum cost
    visited = {}
    
    while pq:
        cost, pos, path = heapq.heappop(pq)
        
        # Skip if already visited with lower cost
        if pos in visited and visited[pos] <= cost:
            continue
        
        visited[pos] = cost
        
        # Check if reached end
        if pos == n:
            return path, cost
        
        # Explore neighbors
        for edge in graph.get_edges_from(pos):
            new_pos = edge.end_block
            new_cost = cost + edge.cost
            
            if new_pos not in visited or visited[new_pos] > new_cost:
                # Get best character
                if edge.predictions:
                    best_pred = edge.predictions[0]
                    character = best_pred.get('character', '?')
                    confidence = best_pred.get('confidence', 0.0)
                else:
                    character = '?'
                    confidence = 0.0
                
                path_node = PathNode(
                    edge=edge,
                    cumulative_cost=new_cost,
                    character=character,
                    confidence=confidence
                )
                
                heapq.heappush(pq, (new_cost, new_pos, path + [path_node]))
    
    # No path found
    return [], float('inf')


def build_recognition_result(
    path: List[PathNode],
    total_cost: float,
    num_blocks: int
) -> RecognitionResult:
    """
    Build the final recognition result from the optimal path
    
    Args:
        path: Optimal path (list of PathNode)
        total_cost: Total cost of the path
        num_blocks: Total number of atomic blocks
        
    Returns:
        RecognitionResult object
    """
    text = ''.join(node.character for node in path)
    
    characters = []
    for node in path:
        # Handle potential infinity or NaN values
        cost_val = node.edge.cost
        if cost_val == float('inf') or cost_val != cost_val:  # inf or NaN
            cost_val = 999.0
        
        char_info = {
            'character': node.character,
            'confidence': float(node.confidence),
            'x_start': int(node.edge.x_start),
            'x_end': int(node.edge.x_end),
            'y_start': int(node.edge.y_start),
            'y_end': int(node.edge.y_end),
            'width': int(node.edge.width),
            'height': int(node.edge.height),
            'num_blocks': int(node.edge.num_blocks),
            'cost': float(cost_val),
            'predictions': node.edge.predictions[:5] if node.edge.predictions else []
        }
        characters.append(char_info)
    
    # Handle infinity or NaN for total_cost
    safe_total_cost = total_cost
    if safe_total_cost == float('inf') or safe_total_cost != safe_total_cost:  # inf or NaN
        safe_total_cost = 999.0
    
    return RecognitionResult(
        text=text,
        characters=characters,
        total_cost=safe_total_cost,
        num_blocks=num_blocks,
        num_characters=len(path)
    )


def search_optimal_path(
    graph: CandidateGraph,
    method: str = 'dp',
    shape_penalty_weight: float = 0.1
) -> RecognitionResult:
    """
    Search for the optimal segmentation path
    
    Args:
        graph: Candidate graph with predictions filled in
        method: Search method ('dp' or 'dijkstra')
        shape_penalty_weight: Weight for shape penalty
        
    Returns:
        RecognitionResult object
    """
    if method == 'dijkstra':
        path, total_cost = dijkstra_search(graph, shape_penalty_weight)
    else:
        path, total_cost = dp_search(graph, shape_penalty_weight)
    
    return build_recognition_result(path, total_cost, graph.num_blocks)


def fill_graph_with_predictions(
    graph: CandidateGraph,
    batch_predictions: List[List[Dict]]
) -> None:
    """
    Fill the graph edges with model predictions
    
    Args:
        graph: Candidate graph
        batch_predictions: List of prediction lists for each edge
    """
    for idx, edge in enumerate(graph.edges):
        if idx < len(batch_predictions):
            edge.predictions = batch_predictions[idx]
            edge.cost = compute_edge_cost(edge.predictions, edge=edge)


class PathSearcher:
    """
    Path searcher class that encapsulates the search logic
    """
    
    def __init__(
        self,
        method: str = 'dp',
        shape_penalty_weight: float = 0.1,
        confidence_threshold: float = 0.01
    ):
        """
        Initialize path searcher
        
        Args:
            method: Search method ('dp' or 'dijkstra')
            shape_penalty_weight: Weight for shape penalty
            confidence_threshold: Minimum confidence threshold
        """
        self.method = method
        self.shape_penalty_weight = shape_penalty_weight
        self.confidence_threshold = confidence_threshold
    
    def search(
        self,
        graph: CandidateGraph
    ) -> RecognitionResult:
        """
        Search for optimal path in the graph
        
        Args:
            graph: Candidate graph with predictions filled
            
        Returns:
            RecognitionResult
        """
        return search_optimal_path(
            graph,
            method=self.method,
            shape_penalty_weight=self.shape_penalty_weight
        )
    
    def fill_and_search(
        self,
        graph: CandidateGraph,
        batch_predictions: List[List[Dict]]
    ) -> RecognitionResult:
        """
        Fill graph with predictions and search for optimal path
        
        Args:
            graph: Candidate graph
            batch_predictions: Predictions for each edge
            
        Returns:
            RecognitionResult
        """
        fill_graph_with_predictions(graph, batch_predictions)
        return self.search(graph)


if __name__ == "__main__":
    # Test code
    print("DP Search Module Test")
    
    # Create a simple test graph
    from .candidate_graph import CandidateGraph, CandidateEdge
    
    graph = CandidateGraph(num_blocks=4)
    
    # Add some test edges
    # Edge 0: block 0 -> 1
    edge0 = CandidateEdge(
        start_block=0, end_block=1,
        image=np.zeros((64, 64)),
        width=30, height=40,
        x_start=0, x_end=30
    )
    edge0.predictions = [{'character': 'A', 'confidence': 0.9}]
    graph.add_edge(edge0)
    
    # Edge 1: block 1 -> 2
    edge1 = CandidateEdge(
        start_block=1, end_block=2,
        image=np.zeros((64, 64)),
        width=25, height=40,
        x_start=30, x_end=55
    )
    edge1.predictions = [{'character': 'B', 'confidence': 0.8}]
    graph.add_edge(edge1)
    
    # Edge 2: block 2 -> 3
    edge2 = CandidateEdge(
        start_block=2, end_block=3,
        image=np.zeros((64, 64)),
        width=35, height=40,
        x_start=55, x_end=90
    )
    edge2.predictions = [{'character': 'C', 'confidence': 0.85}]
    graph.add_edge(edge2)
    
    # Edge 3: block 3 -> 4
    edge3 = CandidateEdge(
        start_block=3, end_block=4,
        image=np.zeros((64, 64)),
        width=28, height=40,
        x_start=90, x_end=118
    )
    edge3.predictions = [{'character': 'D', 'confidence': 0.75}]
    graph.add_edge(edge3)
    
    # Edge 4: block 0 -> 2 (two blocks combined)
    edge4 = CandidateEdge(
        start_block=0, end_block=2,
        image=np.zeros((64, 64)),
        width=55, height=40,
        x_start=0, x_end=55
    )
    edge4.predictions = [{'character': 'E', 'confidence': 0.3}]
    graph.add_edge(edge4)
    
    # Search for optimal path
    result = search_optimal_path(graph, method='dp')
    
    print(f"Recognized text: {result.text}")
    print(f"Total cost: {result.total_cost:.4f}")
    print(f"Number of characters: {result.num_characters}")
    print("Character details:")
    for char in result.characters:
        print(f"  '{char['character']}' (confidence: {char['confidence']:.2f}, "
              f"blocks: {char['num_blocks']}, cost: {char['cost']:.4f})")