"""
Utility functions for the NLP research analysis system.
"""

import os
import logging
from typing import List, Dict, Any
import re


def setup_logging(log_file: str = "nlp_system.log") -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Basic text cleaning operations.
    
    Args:
        text: Raw input text
        
    Returns:
        Cleaned text with normalized whitespace and removed special characters
    """
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def validate_file_path(file_path: str, allowed_extensions: List[str] = None) -> bool:
    """
    Validate file path exists and has allowed extension.
    
    Args:
        file_path: Path to file
        allowed_extensions: List of allowed file extensions (e.g., ['.txt', '.pdf'])
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(file_path):
        return False
    
    if allowed_extensions:
        _, ext = os.path.splitext(file_path)
        return ext.lower() in allowed_extensions
    
    return True


def ensure_directory_exists(directory_path: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory_path: Path to directory
    """
    os.makedirs(directory_path, exist_ok=True)


def truncate_text(text: str, max_length: int = 1000) -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum character length
        
    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def format_topic_keywords(keywords: List[tuple], num_words: int = 10) -> str:
    """
    Format topic keywords with weights for display.
    
    Args:
        keywords: List of (word, weight) tuples
        num_words: Number of words to include
        
    Returns:
        Formatted string of keywords with weights
    """
    keywords = keywords[:num_words]
    formatted = []
    for word, weight in keywords:
        formatted.append(f"{word} ({weight:.3f})")
    return ", ".join(formatted)


def calculate_percentage(part: int, whole: int) -> float:
    """
    Calculate percentage safely.
    
    Args:
        part: Numerator
        whole: Denominator
        
    Returns:
        Percentage value (0-100)
    """
    if whole == 0:
        return 0.0
    return (part / whole) * 100
