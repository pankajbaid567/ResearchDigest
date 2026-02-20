"""
Keyword and Theme Extraction Module
Extract keywords and generate theme labels from topics.
"""

from typing import List, Dict, Tuple, Optional
import logging
from collections import Counter
import numpy as np


class KeywordExtractor:
    """
    Extract keywords and themes from topics and documents.
    Uses deterministic, statistical methods only.
    """
    
    def __init__(self):
        """Initialize keyword extractor."""
        self.logger = logging.getLogger(__name__)
    
    def extract_keywords_from_topics(self, topics: Dict[int, List[Tuple[str, float]]],
                                    n_keywords: int = 10) -> Dict[int, List[str]]:
        """
        Extract top keywords from LDA topics.
        
        Args:
            topics: Dictionary mapping topic_id to list of (word, weight) tuples
            n_keywords: Number of keywords to extract per topic
            
        Returns:
            Dictionary mapping topic_id to list of keywords
        """
        keywords = {}
        
        for topic_id, word_weights in topics.items():
            # Extract top N words
            top_words = [word for word, weight in word_weights[:n_keywords]]
            keywords[topic_id] = top_words
        
        return keywords
    
    def extract_keywords_tfidf(self, tfidf_matrix: np.ndarray,
                              feature_names: List[str],
                              n: int = 20) -> List[Tuple[str, float]]:
        """
        Extract overall top keywords using TF-IDF scores.
        
        Args:
            tfidf_matrix: TF-IDF feature matrix
            feature_names: List of feature names
            n: Number of top keywords
            
        Returns:
            List of (keyword, score) tuples
        """
        # Calculate mean TF-IDF across all documents
        mean_scores = np.mean(tfidf_matrix, axis=0)
        
        # Get top N indices
        top_indices = np.argsort(mean_scores)[::-1][:n]
        
        # Extract keywords and scores
        keywords = [
            (feature_names[idx], mean_scores[idx])
            for idx in top_indices
        ]
        
        return keywords
    
    def generate_theme_labels(self, topic_keywords: Dict[int, List[str]],
                            strategy: str = 'simple') -> Dict[int, str]:
        """
        Generate human-readable theme labels for topics.
        
        Note: This is a simple heuristic approach. Classical NLP struggles
        with automatic theme labeling - this is a known limitation.
        
        Args:
            topic_keywords: Dictionary mapping topic_id to keywords
            strategy: Labeling strategy ('simple' or 'combined')
            
        Returns:
            Dictionary mapping topic_id to theme label
        """
        theme_labels = {}
        
        for topic_id, keywords in topic_keywords.items():
            if strategy == 'simple':
                # Simple: use top 3 keywords
                label = " + ".join(keywords[:3])
            elif strategy == 'combined':
                # Try to find common theme (very basic)
                label = self._infer_theme(keywords)
            else:
                label = f"Topic {topic_id}"
            
            theme_labels[topic_id] = label
        
        return theme_labels
    
    def _infer_theme(self, keywords: List[str]) -> str:
        """
        Attempt to infer theme from keywords (basic heuristic).
        
        This is intentionally simple to demonstrate the limitations
        of classical NLP in semantic understanding.
        
        Args:
            keywords: List of topic keywords
            
        Returns:
            Inferred theme label
        """
        # Predefined theme patterns (domain-specific, limited)
        theme_patterns = {
            'research': ['research', 'study', 'analysis', 'method', 'data', 'experiment'],
            'learning': ['learn', 'learning', 'knowledge', 'education', 'training', 'skill'],
            'technology': ['technology', 'system', 'software', 'algorithm', 'computer', 'digital'],
            'health': ['health', 'medical', 'patient', 'treatment', 'disease', 'clinical'],
            'business': ['business', 'market', 'company', 'financial', 'economic', 'management'],
            'social': ['social', 'people', 'community', 'society', 'cultural', 'human']
        }
        
        # Count matches for each theme
        theme_scores = {}
        keywords_lower = [k.lower() for k in keywords]
        
        for theme, pattern_words in theme_patterns.items():
            score = sum(1 for kw in keywords_lower if any(pw in kw for pw in pattern_words))
            if score > 0:
                theme_scores[theme] = score
        
        # Return best matching theme or default
        if theme_scores:
            best_theme = max(theme_scores, key=theme_scores.get)
            return f"{best_theme.title()}: {' + '.join(keywords[:3])}"
        else:
            return " + ".join(keywords[:3])
    
    def extract_document_keywords(self, tfidf_matrix: np.ndarray,
                                 feature_names: List[str],
                                 doc_idx: int,
                                 n: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords for a specific document.
        
        Args:
            tfidf_matrix: TF-IDF matrix
            feature_names: Feature names
            doc_idx: Document index
            n: Number of keywords
            
        Returns:
            List of (keyword, score) tuples
        """
        # Get document vector
        doc_vector = tfidf_matrix[doc_idx]
        
        # Get top N indices
        top_indices = np.argsort(doc_vector)[::-1][:n]
        
        # Extract keywords
        keywords = [
            (feature_names[idx], doc_vector[idx])
            for idx in top_indices if doc_vector[idx] > 0
        ]
        
        return keywords
    
    def get_keyword_frequencies(self, documents: List[List[str]]) -> Dict[str, int]:
        """
        Calculate keyword frequencies across all documents.
        
        Args:
            documents: List of tokenized documents
            
        Returns:
            Dictionary mapping keywords to frequencies
        """
        # Flatten all tokens
        all_tokens = [token for doc in documents for token in doc]
        
        # Count frequencies
        freq_dist = Counter(all_tokens)
        
        return dict(freq_dist)
    
    def get_top_keywords_by_frequency(self, documents: List[List[str]],
                                     n: int = 20) -> List[Tuple[str, int]]:
        """
        Get top keywords by raw frequency.
        
        Args:
            documents: List of tokenized documents
            n: Number of keywords
            
        Returns:
            List of (keyword, frequency) tuples
        """
        freq_dist = self.get_keyword_frequencies(documents)
        
        # Sort by frequency
        top_keywords = sorted(freq_dist.items(), key=lambda x: x[1], reverse=True)[:n]
        
        return top_keywords
