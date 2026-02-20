"""
Feature Extraction Module
TF-IDF and Bag-of-Words vectorization using scikit-learn.
"""

from typing import List, Dict, Tuple, Optional
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class FeatureExtractor:
    """
    Extract features from preprocessed text using classical methods.
    Primary: TF-IDF
    Optional: Bag-of-Words
    """
    
    def __init__(self):
        """Initialize feature extractor."""
        self.logger = logging.getLogger(__name__)
        self.tfidf_vectorizer = None
        self.bow_vectorizer = None
        self.feature_names = None
    
    def extract_tfidf(self, documents: List[str], max_features: int = 500,
                     ngram_range: Tuple[int, int] = (1, 2),
                     min_df: int = 2, max_df: float = 0.85) -> Tuple[np.ndarray, List[str]]:
        """
        Extract TF-IDF features from documents.
        
        Args:
            documents: List of preprocessed documents (as strings)
            max_features: Maximum number of features to extract
            ngram_range: Range of n-grams to consider (min_n, max_n)
            min_df: Minimum document frequency
            max_df: Maximum document frequency (as fraction)
            
        Returns:
            Tuple of (TF-IDF matrix, feature names)
        """
        try:
            # Adapt parameters for small document collections
            num_docs = len(documents)
            
            # For small collections, relax constraints
            if num_docs == 1:
                # Single document: no filtering needed
                adaptive_min_df = 1
                adaptive_max_df = 1.0
            elif num_docs <= 3:
                # Very small collection
                adaptive_min_df = 1
                adaptive_max_df = 1.0  # Don't filter by max frequency
            else:
                # Normal collection - use provided values
                adaptive_min_df = min(min_df, num_docs // 2)
                adaptive_max_df = max_df
            
            self.logger.info(f"TF-IDF: {num_docs} docs, min_df={adaptive_min_df}, max_df={adaptive_max_df}")
            
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=adaptive_min_df,
                max_df=adaptive_max_df,
                sublinear_tf=True,  # Use log-scaled frequencies
                use_idf=True,
                smooth_idf=True
            )
            
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            self.feature_names = list(feature_names)
            
            self.logger.info(f"Extracted TF-IDF features: {tfidf_matrix.shape}")
            self.logger.info(f"Feature names: {len(feature_names)}")
            
            return tfidf_matrix.toarray(), self.feature_names
            
        except Exception as e:
            self.logger.error(f"TF-IDF extraction failed: {str(e)}")
            raise
    
    def extract_bow(self, documents: List[str], max_features: int = 500,
                   ngram_range: Tuple[int, int] = (1, 1),
                   min_df: int = 2, max_df: float = 0.85) -> Tuple[np.ndarray, List[str]]:
        """
        Extract Bag-of-Words features from documents.
        
        Args:
            documents: List of preprocessed documents (as strings)
            max_features: Maximum number of features
            ngram_range: Range of n-grams
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            
        Returns:
            Tuple of (BoW matrix, feature names)
        """
        try:
            # Adapt parameters for small document collections
            num_docs = len(documents)
            
            if num_docs == 1:
                adaptive_min_df = 1
                adaptive_max_df = 1.0
            elif num_docs <= 3:
                adaptive_min_df = 1
                adaptive_max_df = 1.0
            else:
                adaptive_min_df = min(min_df, num_docs // 2)
                adaptive_max_df = max_df
            
            self.bow_vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=adaptive_min_df,
                max_df=adaptive_max_df
            )
            
            bow_matrix = self.bow_vectorizer.fit_transform(documents)
            feature_names = self.bow_vectorizer.get_feature_names_out()
            
            self.logger.info(f"Extracted BoW features: {bow_matrix.shape}")
            
            return bow_matrix.toarray(), list(feature_names)
            
        except Exception as e:
            self.logger.error(f"BoW extraction failed: {str(e)}")
            raise
    
    def get_top_terms_per_document(self, tfidf_matrix: np.ndarray, 
                                   feature_names: List[str],
                                   n: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get top-N terms for each document based on TF-IDF scores.
        
        Args:
            tfidf_matrix: TF-IDF matrix (documents x features)
            feature_names: List of feature names
            n: Number of top terms to extract
            
        Returns:
            Dictionary mapping document index to list of (term, score) tuples
        """
        top_terms = {}
        
        for doc_idx in range(tfidf_matrix.shape[0]):
            # Get TF-IDF scores for this document
            scores = tfidf_matrix[doc_idx]
            
            # Get top N indices
            top_indices = np.argsort(scores)[::-1][:n]
            
            # Extract terms and scores
            terms_scores = [
                (feature_names[idx], scores[idx]) 
                for idx in top_indices if scores[idx] > 0
            ]
            
            top_terms[doc_idx] = terms_scores
        
        return top_terms
    
    def get_overall_top_terms(self, tfidf_matrix: np.ndarray,
                             feature_names: List[str],
                             n: int = 20) -> List[Tuple[str, float]]:
        """
        Get overall top terms across all documents.
        
        Args:
            tfidf_matrix: TF-IDF matrix
            feature_names: List of feature names
            n: Number of top terms
            
        Returns:
            List of (term, average_score) tuples
        """
        # Calculate mean TF-IDF score for each term
        mean_scores = np.mean(tfidf_matrix, axis=0)
        
        # Get top N indices
        top_indices = np.argsort(mean_scores)[::-1][:n]
        
        # Extract terms and scores
        top_terms = [
            (feature_names[idx], mean_scores[idx])
            for idx in top_indices
        ]
        
        return top_terms
    
    def transform_new_documents(self, documents: List[str], 
                               method: str = 'tfidf') -> np.ndarray:
        """
        Transform new documents using fitted vectorizer.
        
        Args:
            documents: List of documents to transform
            method: 'tfidf' or 'bow'
            
        Returns:
            Transformed feature matrix
        """
        if method == 'tfidf':
            if self.tfidf_vectorizer is None:
                raise ValueError("TF-IDF vectorizer not fitted. Call extract_tfidf first.")
            return self.tfidf_vectorizer.transform(documents).toarray()
        elif method == 'bow':
            if self.bow_vectorizer is None:
                raise ValueError("BoW vectorizer not fitted. Call extract_bow first.")
            return self.bow_vectorizer.transform(documents).toarray()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def get_feature_statistics(self, tfidf_matrix: np.ndarray) -> Dict[str, float]:
        """
        Calculate feature statistics.
        
        Args:
            tfidf_matrix: TF-IDF matrix
            
        Returns:
            Dictionary of statistics
        """
        return {
            'num_documents': tfidf_matrix.shape[0],
            'num_features': tfidf_matrix.shape[1],
            'sparsity': 1.0 - (np.count_nonzero(tfidf_matrix) / tfidf_matrix.size),
            'mean_tfidf': np.mean(tfidf_matrix),
            'max_tfidf': np.max(tfidf_matrix)
        }
