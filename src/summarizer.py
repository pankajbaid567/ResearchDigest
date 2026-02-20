"""
Extractive Summarization Module  
TF-IDF and frequency-based sentence extraction (no generative methods).
"""

from typing import List, Dict, Tuple, Optional
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re


class ExtractiveSummarizer:
    """
    Generate extractive summaries using classical NLP techniques.
    Methods: TF-IDF sentence scoring, frequency-based ranking.
    """
    
    def __init__(self):
        """Initialize summarizer."""
        self.logger = logging.getLogger(__name__)
    
    def summarize_tfidf(self, text: str, num_sentences: int = 5,
                       min_sentence_length: int = 10) -> str:
        """
        Generate extractive summary using TF-IDF sentence scoring.
        
        Args:
            text: Input text to summarize
            num_sentences: Number of sentences in summary
            min_sentence_length: Minimum sentence length (words)
            
        Returns:
            Extractive summary
        """
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Filter short sentences
        valid_sentences = [s for s in sentences if len(s.split()) >= min_sentence_length]
        
        if len(valid_sentences) <= num_sentences:
            return ' '.join(valid_sentences)
        
        # Calculate TF-IDF for sentences
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        
        try:
            tfidf_matrix = vectorizer.fit_transform(valid_sentences)
            
            # Calculate sentence scores (sum of TF-IDF values)
            sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
            
            # Get top sentences
            top_indices = np.argsort(sentence_scores)[::-1][:num_sentences]
            
            # Sort by original order to maintain coherence
            top_indices = sorted(top_indices)
            
            # Extract summary sentences
            summary_sentences = [valid_sentences[idx] for idx in top_indices]
            
            summary = ' '.join(summary_sentences)
            
            self.logger.info(f"Generated summary: {len(summary_sentences)} sentences")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"TF-IDF summarization failed: {str(e)}")
            # Fallback: return first N sentences
            return ' '.join(valid_sentences[:num_sentences])
    
    def summarize_frequency(self, text: str, num_sentences: int = 5,
                           min_sentence_length: int = 10) -> str:
        """
        Generate extractive summary using word frequency-based scoring.
        
        Args:
            text: Input text
            num_sentences: Number of sentences in summary
            min_sentence_length: Minimum sentence length
            
        Returns:
            Extractive summary
        """
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Filter short sentences
        valid_sentences = [s for s in sentences if len(s.split()) >= min_sentence_length]
        
        if len(valid_sentences) <= num_sentences:
            return ' '.join(valid_sentences)
        
        # Tokenize and calculate word frequencies
        words = self._tokenize_words(text)
        word_freq = Counter(words)
        
        # Normalize frequencies
        max_freq = max(word_freq.values()) if word_freq else 1
        for word in word_freq:
            word_freq[word] = word_freq[word] / max_freq
        
        # Score sentences
        sentence_scores = {}
        for idx, sentence in enumerate(valid_sentences):
            score = self._score_sentence(sentence, word_freq)
            sentence_scores[idx] = score
        
        # Get top sentences
        top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
        
        # Sort by original order
        top_indices = sorted(top_indices)
        
        # Extract summary
        summary_sentences = [valid_sentences[idx] for idx in top_indices]
        summary = ' '.join(summary_sentences)
        
        self.logger.info(f"Generated frequency-based summary: {len(summary_sentences)} sentences")
        
        return summary
    
    def summarize_position_weighted(self, text: str, num_sentences: int = 5,
                                    min_sentence_length: int = 10,
                                    position_weight: float = 0.3) -> str:
        """
        Extractive summary with position weighting (first sentences weighted higher).
        
        Args:
            text: Input text
            num_sentences: Number of sentences
            min_sentence_length: Minimum sentence length
            position_weight: Weight for position bonus (0-1)
            
        Returns:
            Extractive summary
        """
        sentences = self._split_sentences(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        valid_sentences = [s for s in sentences if len(s.split()) >= min_sentence_length]
        
        if len(valid_sentences) <= num_sentences:
            return ' '.join(valid_sentences)
        
        # Calculate TF-IDF scores
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        
        try:
            tfidf_matrix = vectorizer.fit_transform(valid_sentences)
            content_scores = np.sum(tfidf_matrix.toarray(), axis=1)
            
            # Normalize content scores
            content_scores = content_scores / np.max(content_scores)
            
            # Add position bonus (earlier sentences get higher scores)
            position_scores = np.array([
                1.0 - (i / len(valid_sentences)) for i in range(len(valid_sentences))
            ])
            
            # Combine scores
            final_scores = (1 - position_weight) * content_scores + position_weight * position_scores
            
            # Get top sentences
            top_indices = np.argsort(final_scores)[::-1][:num_sentences]
            top_indices = sorted(top_indices)
            
            summary_sentences = [valid_sentences[idx] for idx in top_indices]
            
            return ' '.join(summary_sentences)
            
        except Exception as e:
            self.logger.error(f"Position-weighted summarization failed: {str(e)}")
            return ' '.join(valid_sentences[:num_sentences])
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting using regex
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of words
        """
        # Remove special characters and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # Filter stopwords (basic set)
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
            'she', 'it', 'we', 'they', 'them', 'their', 'our'
        }
        
        words = [w for w in words if w not in stopwords and len(w) > 2]
        
        return words
    
    def _score_sentence(self, sentence: str, word_freq: Dict[str, float]) -> float:
        """
        Calculate sentence score based on word frequencies.
        
        Args:
            sentence: Input sentence
            word_freq: Word frequency dictionary
            
        Returns:
            Sentence score
        """
        words = self._tokenize_words(sentence)
        
        if not words:
            return 0.0
        
        # Sum of word frequencies
        score = sum(word_freq.get(word, 0) for word in words)
        
        # Normalize by sentence length
        score = score / len(words)
        
        return score
    
    def get_summary_statistics(self, original_text: str, summary: str) -> Dict[str, any]:
        """
        Calculate summary statistics.
        
        Args:
            original_text: Original text
            summary: Generated summary
            
        Returns:
            Dictionary of statistics
        """
        original_sentences = self._split_sentences(original_text)
        summary_sentences = self._split_sentences(summary)
        
        original_words = len(original_text.split())
        summary_words = len(summary.split())
        
        return {
            'original_sentences': len(original_sentences),
            'summary_sentences': len(summary_sentences),
            'original_words': original_words,
            'summary_words': summary_words,
            'compression_ratio': summary_words / original_words if original_words > 0 else 0,
            'sentence_retention': len(summary_sentences) / len(original_sentences) if original_sentences else 0
        }
