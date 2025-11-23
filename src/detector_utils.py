"""
Utility functions for detecting watermarked code.

This module provides a simple interface for checking if Python code
is watermarked using trained Bayesian detectors.
"""

import pickle
import io
import torch
import os
from pathlib import Path
from transformers import AutoTokenizer
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables
load_dotenv()


class CPU_Unpickler(pickle.Unpickler):
    """Custom unpickler to load CUDA models on CPU and handle old module paths."""
    def find_class(self, module, name):
        # Handle CUDA to CPU mapping
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        
        # Remap old module paths to new src structure
        # Models were pickled when modules were in root, now they're in src/
        if module == 'bayesian_detector':
            module = 'src.bayesian_detector'
        elif module in ['model_utils', 'execution_utils', 'report_generator']:
            module = f'src.{module}'
        
        return super().find_class(module, name)


class WatermarkDetector:
    """Simple interface for detecting watermarked Python code."""
    
    def __init__(self, model_path=None, ngram_len=5):
        """
        Initialize the watermark detector.
        
        Args:
            model_path: Path to the detector .pkl file. If None, uses default location.
            ngram_len: Which ngram detector to use (2, 5, or 10). Default is 5.
        """
        self.ngram_len = ngram_len
        
        # Default to outputs/models directory
        if model_path is None:
            repo_root = Path(__file__).parent.parent
            model_path = repo_root / f"outputs/models/bayesian_detector_ngram{ngram_len}.pkl"
        
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Detector model not found at {self.model_path}. "
                f"Please train the detector first with: python scripts/train_bayesian_detector.py --train"
            )
        
        # Load detector
        print(f"Loading detector from {self.model_path}...")
        with open(self.model_path, 'rb') as f:
            saved_data = CPU_Unpickler(f).load()
            self.detector = saved_data['detector']
        
        # Force CPU device to avoid CUDA errors
        if hasattr(self.detector, 'logits_processor'):
            self.detector.logits_processor.device = torch.device('cpu')
        
        # Login to Hugging Face if token is available
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token, add_to_git_credential=False)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("google/codegemma-7b-it")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Detector loaded successfully (ngram_len={ngram_len}, device={self.device})")
    
    def is_watermarked(self, code: str, threshold: float = 0.5) -> bool:
        """
        Check if Python code is watermarked.
        
        Args:
            code: String containing Python code to check
            threshold: Score threshold for considering code watermarked (0-1).
                      Default is 0.5. Higher threshold = more confident detection.
        
        Returns:
            True if code appears to be watermarked, False otherwise
        """
        score = self.get_score(code)
        return score >= threshold
    
    def get_score(self, code: str) -> float:
        """
        Get the watermark detection score for Python code.
        
        Args:
            code: String containing Python code to check
        
        Returns:
            Float score between 0 and 1, where higher values indicate
            more confidence that the code is watermarked.
        """
        if not code or not code.strip():
            return 0.0
        
        try:
            # Tokenize the code
            tokens = self.tokenizer.encode(
                code, 
                return_tensors='pt', 
                truncation=True, 
                max_length=2048
            )
            tokens = tokens.to(self.device)
            
            # Get detection score
            score = float(self.detector.score(tokens)[0])
            return score
            
        except Exception as e:
            print(f"Error scoring code: {e}")
            return 0.0
    
    def detect(self, code: str, threshold: float = 0.5) -> dict:
        """
        Detect watermark and return detailed results.
        
        Args:
            code: String containing Python code to check
            threshold: Score threshold for watermark detection (0-1)
        
        Returns:
            Dictionary with detection results:
            {
                'is_watermarked': bool,
                'score': float,
                'confidence': str,  # 'low', 'medium', 'high'
                'threshold': float
            }
        """
        score = self.get_score(code)
        is_watermarked = score >= threshold
        
        # Determine confidence level
        if score < 0.3:
            confidence = 'low'
        elif score < 0.7:
            confidence = 'medium'
        else:
            confidence = 'high'
        
        return {
            'is_watermarked': is_watermarked,
            'score': score,
            'confidence': confidence,
            'threshold': threshold,
            'ngram_len': self.ngram_len
        }


def detect_watermark(code: str, ngram_len: int = 5, threshold: float = 0.5) -> bool:
    """
    Simple function to check if Python code is watermarked.
    
    This is a convenience function that creates a detector and checks the code.
    For multiple detections, it's more efficient to create a WatermarkDetector
    instance and reuse it.
    
    Args:
        code: String containing Python code to check
        ngram_len: Which ngram detector to use (2, 5, or 10). Default is 5.
        threshold: Score threshold for watermark detection (0-1). Default is 0.5.
    
    Returns:
        True if code appears to be watermarked, False otherwise
    
    Example:
        >>> code = "def hello():\\n    print('Hello, world!')"
        >>> is_watermarked = detect_watermark(code)
        >>> print(f"Watermarked: {is_watermarked}")
    """
    detector = WatermarkDetector(ngram_len=ngram_len)
    return detector.is_watermarked(code, threshold=threshold)


def get_watermark_score(code: str, ngram_len: int = 5) -> float:
    """
    Get the watermark detection score for Python code.
    
    Args:
        code: String containing Python code to check
        ngram_len: Which ngram detector to use (2, 5, or 10). Default is 5.
    
    Returns:
        Float score between 0 and 1, where higher values indicate
        more confidence that the code is watermarked.
    
    Example:
        >>> code = "def hello():\\n    print('Hello, world!')"
        >>> score = get_watermark_score(code)
        >>> print(f"Watermark score: {score:.4f}")
    """
    detector = WatermarkDetector(ngram_len=ngram_len)
    return detector.get_score(code)
