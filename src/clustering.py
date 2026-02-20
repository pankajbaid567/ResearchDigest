"""
Clustering Module
K-Means clustering as an alternative to LDA for topic discovery.
"""

from typing import List, Dict, Tuple, Optional
import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class DocumentClusterer:
    """
    Perform document clustering using K-Means.
    Alternative approach to LDA for grouping similar documents.
    """
    
    def __init__(self):
        """Initialize clusterer."""
        self.logger = logging.getLogger(__name__)
        self.kmeans_model = None
        self.feature_names = None
    
    def perform_kmeans(self, feature_matrix: np.ndarray, n_clusters: int = 5,
                      random_state: int = 42, max_iter: int = 300) -> KMeans:
        """
        Perform K-Means clustering on feature matrix.
        
        Args:
            feature_matrix: Document feature matrix (e.g., TF-IDF)
            n_clusters: Number of clusters
            random_state: Random seed
            max_iter: Maximum iterations
            
        Returns:
            Fitted KMeans model
        """
        self.logger.info(f"Performing K-Means clustering with {n_clusters} clusters...")
        
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            n_init=10
        )
        
        self.kmeans_model.fit(feature_matrix)
        
        self.logger.info("K-Means clustering complete")
        self.logger.info(f"Inertia: {self.kmeans_model.inertia_:.2f}")
        
        return self.kmeans_model
    
    def find_optimal_clusters(self, feature_matrix: np.ndarray,
                            min_k: int = 2, max_k: int = 10) -> Tuple[int, Dict[int, float]]:
        """
        Find optimal number of clusters using elbow method and silhouette score.
        
        Args:
            feature_matrix: Document feature matrix
            min_k: Minimum number of clusters to try
            max_k: Maximum number of clusters to try
            
        Returns:
            Tuple of (optimal_k, scores_dict)
        """
        inertias = {}
        silhouette_scores = {}
        
        for k in range(min_k, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(feature_matrix)
            
            inertias[k] = kmeans.inertia_
            
            # Calculate silhouette score
            if k > 1:
                sil_score = silhouette_score(feature_matrix, labels)
                silhouette_scores[k] = sil_score
        
        # Find optimal K (highest silhouette score)
        optimal_k = max(silhouette_scores, key=silhouette_scores.get)
        
        self.logger.info(f"Optimal number of clusters: {optimal_k}")
        self.logger.info(f"Silhouette score: {silhouette_scores[optimal_k]:.4f}")
        
        return optimal_k, {'inertias': inertias, 'silhouette_scores': silhouette_scores}
    
    def get_cluster_labels(self) -> np.ndarray:
        """
        Get cluster labels for documents.
        
        Returns:
            Array of cluster labels
        """
        if self.kmeans_model is None:
            raise ValueError("Model not trained. Call perform_kmeans first.")
        
        return self.kmeans_model.labels_
    
    def get_top_terms_per_cluster(self, tfidf_matrix: np.ndarray,
                                  feature_names: List[str],
                                  n_terms: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """
        Extract top terms for each cluster based on cluster centroids.
        
        Args:
            tfidf_matrix: TF-IDF feature matrix
            feature_names: List of feature names
            n_terms: Number of top terms per cluster
            
        Returns:
            Dictionary mapping cluster_id to list of (term, score) tuples
        """
        if self.kmeans_model is None:
            raise ValueError("Model not trained.")
        
        self.feature_names = feature_names
        cluster_terms = {}
        
        # Get cluster centers
        centers = self.kmeans_model.cluster_centers_
        
        for cluster_id in range(len(centers)):
            # Get centroid for this cluster
            centroid = centers[cluster_id]
            
            # Get top term indices
            top_indices = np.argsort(centroid)[::-1][:n_terms]
            
            # Extract terms and scores
            terms_scores = [
                (feature_names[idx], centroid[idx])
                for idx in top_indices
            ]
            
            cluster_terms[cluster_id] = terms_scores
        
        return cluster_terms
    
    def get_documents_per_cluster(self, labels: Optional[np.ndarray] = None) -> Dict[int, List[int]]:
        """
        Get document indices for each cluster.
        
        Args:
            labels: Cluster labels (uses model labels if None)
            
        Returns:
            Dictionary mapping cluster_id to list of document indices
        """
        if labels is None:
            labels = self.get_cluster_labels()
        
        clusters = {}
        
        for doc_idx, cluster_id in enumerate(labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(doc_idx)
        
        return clusters
    
    def predict_cluster(self, new_documents: np.ndarray) -> np.ndarray:
        """
        Predict cluster assignments for new documents.
        
        Args:
            new_documents: Feature matrix for new documents
            
        Returns:
            Array of predicted cluster labels
        """
        if self.kmeans_model is None:
            raise ValueError("Model not trained.")
        
        return self.kmeans_model.predict(new_documents)
    
    def get_cluster_statistics(self) -> Dict[int, Dict[str, int]]:
        """
        Get statistics for each cluster.
        
        Returns:
            Dictionary with cluster statistics
        """
        labels = self.get_cluster_labels()
        doc_per_cluster = self.get_documents_per_cluster(labels)
        
        stats = {}
        for cluster_id, doc_indices in doc_per_cluster.items():
            stats[cluster_id] = {
                'num_documents': len(doc_indices),
                'percentage': (len(doc_indices) / len(labels)) * 100
            }
        
        return stats
