"""
Multi-Character Recognition Test Script
Tests the Atomic-DP multi-character recognition pipeline
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import cv2
from recognizers.multi_char import MultiCharHandwritingRecognizer
from utils.atomic_segmentation import segment_image_to_atomic_blocks, visualize_atomic_blocks
from utils.candidate_graph import build_candidate_graph, visualize_candidate_graph


def create_test_image(text: str = "ABC", width: int = 300, height: int = 60) -> np.ndarray:
    """
    Create a simple test image with text
    
    Args:
        text: Text to draw
        width: Image width
        height: Image height
        
    Returns:
        Test image
    """
    # Create white background
    image = np.ones((height, width), dtype=np.uint8) * 255
    
    # Draw text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 2
    
    # Calculate text position
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x = (width - text_size[0]) // 2
    y = (height + text_size[1]) // 2
    
    cv2.putText(image, text, (x, y), font, font_scale, 0, thickness)
    
    return image


def test_atomic_segmentation():
    """Test atomic segmentation module"""
    print("\n" + "="*60)
    print("Testing Atomic Segmentation")
    print("="*60)
    
    # Create test image
    image = create_test_image("ABC")
    
    # Segment
    blocks = segment_image_to_atomic_blocks(image)
    
    print(f"Created test image: {image.shape}")
    print(f"Found {len(blocks)} atomic blocks")
    
    for i, block in enumerate(blocks):
        print(f"  Block {i}: x=[{block.x_start}, {block.x_end}], "
              f"y=[{block.y_start}, {block.y_end}], "
              f"size={block.width}x{block.height}")
    
    # Visualize
    vis = visualize_atomic_blocks(image, blocks, "test_atomic_blocks.png")
    print("Visualization saved to test_atomic_blocks.png")
    
    return blocks


def test_candidate_graph(blocks, original_image):
    """Test candidate graph building"""
    print("\n" + "="*60)
    print("Testing Candidate Graph Building")
    print("="*60)
    
    # Build graph
    graph = build_candidate_graph(blocks, original_image)
    
    print(f"Built graph with {len(graph.edges)} candidate edges")
    
    for i, edge in enumerate(graph.edges):
        print(f"  Edge {i}: blocks [{edge.start_block}, {edge.end_block}), "
              f"size={edge.width}x{edge.height}")
    
    # Visualize
    vis = visualize_candidate_graph(graph, blocks, original_image, "test_candidate_graph.png")
    print("Visualization saved to test_candidate_graph.png")
    
    return graph


def test_multi_char_recognizer():
    """Test the full multi-character recognizer"""
    print("\n" + "="*60)
    print("Testing Multi-Character Recognizer")
    print("="*60)
    
    # Check if model exists
    model_path = Path(__file__).parent / "checkpoints" / "best_hanzi_tiny.pth"
    class_path = Path(__file__).parent / "checkpoints" / "classes.json"
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Skipping recognizer test")
        return None
    
    # Create recognizer
    recognizer = MultiCharHandwritingRecognizer(
        str(model_path),
        config={
            "class_mapping_path": str(class_path),
            "max_blocks_per_char": 3,
            "shape_penalty_weight": 0.1,
            "search_method": "dp"
        }
    )
    
    # Create test image
    # Note: This is a simple test with ASCII characters
    # For real Chinese characters, use actual handwriting images
    image = create_test_image("ABC", width=300, height=60)
    
    # Recognize
    result = recognizer.predict(image, top_k=5)
    
    print(f"Recognition result:")
    print(f"  Text: {result.get('text', '')}")
    print(f"  Success: {result.get('success', False)}")
    print(f"  Num characters: {result.get('num_characters', 0)}")
    print(f"  Num blocks: {result.get('num_blocks', 0)}")
    print(f"  Total cost: {result.get('total_cost', 0):.4f}")
    
    timing = result.get('timing', {})
    print(f"  Timing:")
    for key, value in timing.items():
        print(f"    {key}: {value*1000:.2f}ms")
    
    if result.get('characters'):
        print(f"  Character details:")
        for char in result['characters']:
            print(f"    '{char['character']}' (conf: {char['confidence']:.2f}, "
                  f"blocks: {char['num_blocks']}, x: [{char['x_start']}, {char['x_end']}]")
    
    return result


def test_with_real_image(image_path: str):
    """Test with a real handwriting image"""
    print("\n" + "="*60)
    print(f"Testing with real image: {image_path}")
    print("="*60)
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    print(f"Loaded image: {image.shape}")
    
    # Check if model exists
    model_path = Path(__file__).parent / "checkpoints" / "best_hanzi_tiny.pth"
    class_path = Path(__file__).parent / "checkpoints" / "classes.json"
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return None
    
    # Create recognizer
    recognizer = MultiCharHandwritingRecognizer(
        str(model_path),
        config={
            "class_mapping_path": str(class_path),
            "max_blocks_per_char": 3,
            "shape_penalty_weight": 0.1,
            "search_method": "dp"
        }
    )
    
    # Recognize
    result = recognizer.predict(image, top_k=5)
    
    print(f"Recognition result:")
    print(f"  Text: {result.get('text', '')}")
    print(f"  Success: {result.get('success', False)}")
    print(f"  Num characters: {result.get('num_characters', 0)}")
    print(f"  Num blocks: {result.get('num_blocks', 0)}")
    
    timing = result.get('timing', {})
    total_time = sum(timing.values())
    print(f"  Total time: {total_time*1000:.2f}ms")
    
    return result


def main():
    """Main test function"""
    print("="*60)
    print("Multi-Character Recognition Test Suite")
    print("="*60)
    
    # Test 1: Atomic segmentation
    blocks = test_atomic_segmentation()
    
    if blocks:
        # Test 2: Candidate graph
        image = create_test_image("ABC")
        graph = test_candidate_graph(blocks, image)
    
    # Test 3: Full recognizer
    result = test_multi_char_recognizer()
    
    # Test with real image if provided
    if len(sys.argv) > 1:
        test_with_real_image(sys.argv[1])
    
    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)


if __name__ == "__main__":
    main()