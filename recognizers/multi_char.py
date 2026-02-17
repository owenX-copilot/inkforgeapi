"""
Multi-Character Handwriting Recognizer
Implements the Atomic-DP approach for recognizing multiple handwritten characters

Pipeline:
1. Atomic Segmentation - Split image into atomic blocks
2. Candidate Graph Building - Generate all possible character combinations
3. Batch Recognition - Use single-char model to score all candidates
4. DP Path Search - Find optimal segmentation path
"""

import json
import torch
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import sys
import time
import base64
from io import BytesIO
from PIL import Image

# Add model directory to path
sys.path.append(str(Path(__file__).parent.parent))

from model.hanzi_tiny import HanziTiny
from .base import BaseRecognizer
from utils.atomic_segmentation import segment_image_to_atomic_blocks, AtomicBlock
from utils.candidate_graph import build_candidate_graph, prepare_batch_for_inference, CandidateGraph
from utils.dp_search import PathSearcher, RecognitionResult


class MultiCharHandwritingRecognizer(BaseRecognizer):
    """Multi-character handwriting recognizer using Atomic-DP approach"""
    
    def __init__(self, model_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize multi-character recognizer
        
        Args:
            model_path: Path to the single-character model
            config: Configuration dictionary, can include:
                - class_mapping_path: Path to class mapping file
                - img_size: Image size (default 64)
                - mean: Normalization mean (default 0.5)
                - std: Normalization std (default 0.5)
                - max_blocks_per_char: Max atomic blocks per character (default 3)
                - shape_penalty_weight: Weight for shape penalty (default 0.1)
                - search_method: Path search method ('dp' or 'dijkstra')
        """
        super().__init__(model_path, config)
        
        # Default configuration
        self.config.setdefault("class_mapping_path",
                              str(Path(model_path).parent / "classes.json"))
        self.config.setdefault("img_size", 64)
        self.config.setdefault("mean", 0.5)
        self.config.setdefault("std", 0.5)
        self.config.setdefault("max_blocks_per_char", 3)
        self.config.setdefault("shape_penalty_weight", 0.1)
        self.config.setdefault("search_method", "dp")
        self.config.setdefault("min_gap_threshold", None)  # Auto-calculated
        self.config.setdefault("aspect_ratio_threshold", 1.2)
        
        self.img_size = self.config["img_size"]
        self.mean = self.config["mean"]
        self.std = self.config["std"]
        
        # Class mapping
        self.idx_to_char = {}
        self.char_to_idx = {}
        
        # Load class mapping
        self._load_class_mapping()
        
        # Path searcher
        self.path_searcher = PathSearcher(
            method=self.config["search_method"],
            shape_penalty_weight=self.config["shape_penalty_weight"]
        )
    
    def _load_class_mapping(self) -> None:
        """Load class mapping from file"""
        mapping_path = self.config["class_mapping_path"]
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                encoded_classes = json.load(f)
            
            # Decode Unicode encoding (#U4e00 -> 'One')
            for idx, encoded_char in enumerate(encoded_classes):
                if encoded_char.startswith("#U"):
                    hex_code = encoded_char[2:]
                    char = chr(int(hex_code, 16))
                else:
                    char = encoded_char
                
                self.idx_to_char[idx] = char
                self.char_to_idx[char] = idx
            
            print(f"Loaded class mapping: {len(self.idx_to_char)} characters")
            
        except Exception as e:
            print(f"Failed to load class mapping: {e}")
            # Create default mapping
            for i in range(630):
                self.idx_to_char[i] = f"char_{i}"
                self.char_to_idx[f"char_{i}"] = i
    
    def load_model(self) -> None:
        """Load the single-character model"""
        try:
            num_classes = len(self.idx_to_char)
            self.model = HanziTiny(num_classes=num_classes)
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            
            print(f"Model loaded: {self.model_path}")
            print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"Device: {next(self.model.parameters()).device}")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single image for recognition
        
        Args:
            image: Input image (H, W) or (H, W, C)
            
        Returns:
            Preprocessed tensor [1, 1, img_size, img_size]
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.shape[2] == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            else:
                gray = image.squeeze()
        else:
            gray = image
        
        # Resize
        if gray.shape != (self.img_size, self.img_size):
            gray = cv2.resize(gray, (self.img_size, self.img_size),
                            interpolation=cv2.INTER_AREA)
        
        # Convert to [0, 1] range
        if gray.dtype != np.float32:
            gray = gray.astype(np.float32) / 255.0
        
        # Normalize
        gray = (gray - self.mean) / self.std
        
        # Add batch and channel dimensions [1, 1, H, W]
        tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)
        
        return tensor
    
    def get_class_mapping(self) -> Dict[int, str]:
        """
        Get class mapping
        
        Returns:
            Dictionary mapping index to character
        """
        return self.idx_to_char.copy()
    
    def preprocess_batch(self, batch: np.ndarray) -> torch.Tensor:
        """
        Preprocess a batch of images
        
        Args:
            batch: Batch array of shape (N, 1, H, W) with values in [0, 1]
            
        Returns:
            Preprocessed tensor of shape (N, 1, H, W)
        """
        # Normalize
        batch = (batch - self.mean) / self.std
        
        # Convert to tensor
        tensor = torch.from_numpy(batch).float()
        tensor = tensor.to(self.device)
        
        return tensor
    
    def recognize_batch(
        self,
        batch: np.ndarray,
        top_k: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """
        Recognize a batch of character images
        
        Args:
            batch: Batch array of shape (N, 1, 64, 64)
            top_k: Number of top predictions to return
            
        Returns:
            List of prediction lists for each image
        """
        if batch.shape[0] == 0:
            return []
        
        # Lazy load
        self.lazy_load()
        
        # Preprocess
        input_tensor = self.preprocess_batch(batch)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, k=min(top_k, probabilities.size(1)))
        
        # Convert to results
        all_results = []
        for i in range(batch.shape[0]):
            results = []
            for j in range(top_indices.size(1)):
                idx = top_indices[i, j].item()
                prob = top_probs[i, j].item()
                
                character = self.idx_to_char.get(idx, f"unknown_{idx}")
                
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
            all_results.append(results)
        
        return all_results
    
    def predict(
        self,
        image: np.ndarray,
        top_k: int = 5,
        return_details: bool = True
    ) -> Dict[str, Any]:
        """
        Recognize multiple characters from a handwriting line image
        
        Args:
            image: Input image (can be grayscale or color)
            top_k: Number of top predictions per character
            return_details: Whether to return detailed information
            
        Returns:
            Dictionary containing:
                - text: Recognized text
                - characters: List of character details
                - num_characters: Number of recognized characters
                - num_blocks: Number of atomic blocks
                - total_cost: Total path cost
                - timing: Timing information
        """
        start_time = time.time()
        
        # Lazy load
        self.lazy_load()
        
        timing = {}
        
        # Step 1: Atomic segmentation
        t0 = time.time()
        blocks = segment_image_to_atomic_blocks(
            image,
            min_gap_threshold=self.config.get("min_gap_threshold"),
            aspect_ratio_threshold=self.config.get("aspect_ratio_threshold", 1.2)
        )
        timing['segmentation'] = time.time() - t0
        
        if not blocks:
            return {
                "text": "",
                "characters": [],
                "num_characters": 0,
                "num_blocks": 0,
                "total_cost": 0.0,
                "timing": {k: float(v) for k, v in timing.items()},
                "success": False,
                "error": "No atomic blocks detected"
            }
        
        # Step 2: Build candidate graph
        t0 = time.time()
        graph = build_candidate_graph(
            blocks,
            original_image=image,
            max_blocks_per_char=self.config.get("max_blocks_per_char", 3),
            target_size=self.img_size,
            enable_geometric_filter=True
        )
        timing['graph_building'] = time.time() - t0
        
        print(f"[DEBUG] 原子块数量: {len(blocks)}")
        print(f"[DEBUG] 候选边数量: {len(graph.edges)}")
        for i, edge in enumerate(graph.edges):
            print(f"[DEBUG]   边 {i}: 块[{edge.start_block}->{edge.end_block}], 尺寸={edge.width}x{edge.height}")
        
        if not graph.edges:
            return {
                "text": "",
                "characters": [],
                "num_characters": 0,
                "num_blocks": len(blocks),
                "total_cost": 0.0,
                "timing": {k: float(v) for k, v in timing.items()},
                "success": False,
                "error": "No candidate edges generated"
            }
        
        # Step 3: Prepare batch for inference
        t0 = time.time()
        batch, edge_indices = prepare_batch_for_inference(graph)
        timing['batch_preparation'] = time.time() - t0
        
        if batch.shape[0] == 0:
            return {
                "text": "",
                "characters": [],
                "num_characters": 0,
                "num_blocks": len(blocks),
                "total_cost": 0.0,
                "timing": {k: float(v) for k, v in timing.items()},
                "success": False,
                "error": "Empty batch for inference"
            }
        
        # Step 4: Batch recognition
        t0 = time.time()
        batch_predictions = self.recognize_batch(batch, top_k=top_k)
        timing['recognition'] = time.time() - t0
        
        # Step 5: Fill graph with predictions
        t0 = time.time()
        for idx, edge_idx in enumerate(edge_indices):
            if idx < len(batch_predictions):
                graph.edges[edge_idx].predictions = batch_predictions[idx]
                pred = batch_predictions[idx]
                if pred:
                    print(f"[DEBUG]   边 {edge_idx} 预测: {pred[0].get('character', '?')} (置信度: {pred[0].get('confidence', 0):.3f})")
        timing['graph_filling'] = time.time() - t0
        
        # Step 6: DP path search
        t0 = time.time()
        result = self.path_searcher.search(graph)
        timing['path_search'] = time.time() - t0
        
        print(f"[DEBUG] DP搜索结果: 文本='{result.text}', 字符数={result.num_characters}, 代价={result.total_cost}")
        
        timing['total'] = time.time() - start_time
        
        # Build response
        response = {
            "text": result.text,
            "characters": result.characters,
            "num_characters": result.num_characters,
            "num_blocks": result.num_blocks,
            "total_cost": float(result.total_cost) if result.total_cost != float('inf') else 999.0,
            "timing": {k: float(v) for k, v in timing.items()},
            "success": True
        }
        
        if return_details:
            # Add base64 encoded images for each recognized character
            char_images_base64 = []
            for char_info in result.characters:
                # Find the corresponding edge
                for edge in graph.edges:
                    if (edge.x_start == char_info['x_start'] and 
                        edge.x_end == char_info['x_end']):
                        # Convert edge image to base64
                        img = edge.image
                        if img.dtype != np.uint8:
                            # edge.image is float32 in range [0, 255] from combine_atomic_blocks
                            # Convert to uint8
                            img = img.astype(np.uint8)
                        
                        # Convert to PIL Image and then to base64
                        pil_img = Image.fromarray(img, mode='L')
                        buffer = BytesIO()
                        pil_img.save(buffer, format='PNG')
                        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        
                        char_info['image_base64'] = img_base64
                        break
                
                char_images_base64.append(char_info)
            
            response["characters"] = char_images_base64
            response["atomic_blocks"] = [
                {
                    "x_start": int(b.x_start),
                    "x_end": int(b.x_end),
                    "y_start": int(b.y_start),
                    "y_end": int(b.y_end),
                    "width": int(b.width),
                    "height": int(b.height)
                }
                for b in blocks
            ]
            response["num_candidates"] = len(graph.edges)
        
        return response
    
    def get_info(self) -> Dict[str, Any]:
        """Get recognizer information"""
        info = super().get_info()
        info.update({
            "num_classes": len(self.idx_to_char),
            "img_size": self.img_size,
            "max_blocks_per_char": self.config.get("max_blocks_per_char", 3),
            "search_method": self.config.get("search_method", "dp"),
            "class_mapping_loaded": len(self.idx_to_char) > 0
        })
        return info


def create_multi_char_recognizer(
    model_path: str,
    class_mapping_path: Optional[str] = None,
    **kwargs
) -> MultiCharHandwritingRecognizer:
    """
    Factory function to create a multi-character recognizer
    
    Args:
        model_path: Path to the model file
        class_mapping_path: Path to class mapping file
        **kwargs: Additional configuration options
        
    Returns:
        MultiCharHandwritingRecognizer instance
    """
    config = kwargs.copy()
    if class_mapping_path:
        config["class_mapping_path"] = class_mapping_path
    
    return MultiCharHandwritingRecognizer(model_path, config)


if __name__ == "__main__":
    # Test code
    import sys
    
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
        image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is not None:
            # Create recognizer
            recognizer = MultiCharHandwritingRecognizer(
                model_path="checkpoints/best_hanzi_tiny.pth",
                config={
                    "class_mapping_path": "checkpoints/classes.json"
                }
            )
            
            # Recognize
            result = recognizer.predict(image)
            
            print(f"Recognized text: {result['text']}")
            print(f"Number of characters: {result['num_characters']}")
            print(f"Number of atomic blocks: {result['num_blocks']}")
            print(f"Total cost: {result['total_cost']:.4f}")
            print(f"Timing: {result['timing']}")
            
            if result['characters']:
                print("\nCharacter details:")
                for char in result['characters']:
                    print(f"  '{char['character']}' (confidence: {char['confidence']:.2f}, "
                          f"blocks: {char['num_blocks']}, x: [{char['x_start']}, {char['x_end']}]")
        else:
            print(f"Cannot read image: {test_image_path}")
    else:
        print("Usage: python multi_char.py <image_path>")