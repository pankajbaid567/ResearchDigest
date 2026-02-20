"""
Evaluation Module
Calculate and interpret topic coherence and other metrics.
"""

from typing import Dict, List, Optional, Any
import logging
import numpy as np


class Evaluator:
    """
    Evaluate topic modeling and summarization results.
    Focuses on topic coherence and interpretability.
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.logger = logging.getLogger(__name__)
    
    def evaluate_topics(self, lda_model, texts: List[List[str]], 
                       dictionary, coherence_type: str = 'c_v') -> Dict[str, Any]:
        """
        Comprehensive topic evaluation.
        
        Args:
            lda_model: Trained LDA model
            texts: Tokenized documents
            dictionary: Gensim dictionary
            coherence_type: Type of coherence metric
            
        Returns:
            Dictionary of evaluation metrics
        """
        from gensim.models.coherencemodel import CoherenceModel
        
        # Calculate coherence
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=texts,
            dictionary=dictionary,
            coherence=coherence_type
        )
        
        coherence_score = coherence_model.get_coherence()
        
        # Get per-topic coherence
        coherence_per_topic = coherence_model.get_coherence_per_topic()
        
        # Calculate topic diversity
        diversity = self._calculate_topic_diversity(lda_model)
        
        evaluation = {
            'coherence_score': coherence_score,
            'coherence_type': coherence_type,
            'coherence_per_topic': coherence_per_topic,
            'topic_diversity': diversity,
            'num_topics': lda_model.num_topics,
            'interpretation': self._interpret_coherence(coherence_score, coherence_type)
        }
        
        self.logger.info(f"Coherence Score ({coherence_type}): {coherence_score:.4f}")
        self.logger.info(f"Topic Diversity: {diversity:.4f}")
        
        return evaluation
    
    def _calculate_topic_diversity(self, lda_model, top_n: int = 10) -> float:
        """
        Calculate topic diversity (uniqueness of topics).
        
        Args:
            lda_model: LDA model
            top_n: Number of top words to consider
            
        Returns:
            Diversity score (0-1, higher is better)
        """
        unique_words = set()
        total_words = 0
        
        for topic_id in range(lda_model.num_topics):
            top_words = [word for word, _ in lda_model.show_topic(topic_id, topn=top_n)]
            unique_words.update(top_words)
            total_words += len(top_words)
        
        if total_words == 0:
            return 0.0
        
        diversity = len(unique_words) / total_words
        return diversity
    
    def _interpret_coherence(self, score: float, coherence_type: str) -> str:
        """
        Generate human-readable interpretation of coherence score.
        
        Args:
            score: Coherence score
            coherence_type: Type of coherence
            
        Returns:
            Interpretation string
        """
        if coherence_type == 'c_v':
            # c_v ranges from 0 to 1 (higher is better)
            if score > 0.6:
                quality = "Excellent"
                explanation = "Topics are highly coherent and semantically meaningful."
            elif score > 0.5:
                quality = "Good"
                explanation = "Topics show good coherence with clear themes."
            elif score > 0.4:
                quality = "Moderate"
                explanation = "Topics are somewhat coherent but may have mixed themes."
            elif score > 0.3:
                quality = "Fair"
                explanation = "Topics show weak coherence; consider tuning parameters."
            else:
                quality = "Poor"
                explanation = "Topics lack coherence; significant tuning needed."
                
        elif coherence_type == 'u_mass':
            # u_mass ranges from -14 to 14 (higher is better, typically negative)
            if score > -1:
                quality = "Excellent"
                explanation = "Topics are highly coherent."
            elif score > -2:
                quality = "Good"
                explanation = "Topics show good coherence."
            elif score > -3:
                quality = "Moderate"
                explanation = "Topics are moderately coherent."
            else:
                quality = "Fair to Poor"
                explanation = "Topics show weak coherence."
        else:
            quality = "Unknown"
            explanation = f"Coherence type '{coherence_type}' interpretation not defined."
        
        return f"{quality} ({score:.4f}): {explanation}"
    
    def generate_interpretation(self, topics: Dict[int, List[tuple]], 
                              coherence_score: float) -> str:
        """
        Generate comprehensive interpretation of topic modeling results.
        
        Args:
            topics: Dictionary of topics with words and weights
            coherence_score: Coherence score
            
        Returns:
            Interpretation text
        """
        interpretation = []
        
        interpretation.append("=" * 60)
        interpretation.append("TOPIC MODELING EVALUATION")
        interpretation.append("=" * 60)
        interpretation.append("")
        
        interpretation.append(f"Number of Topics: {len(topics)}")
        interpretation.append(f"Coherence Score (c_v): {coherence_score:.4f}")
        interpretation.append(f"Assessment: {self._interpret_coherence(coherence_score, 'c_v')}")
        interpretation.append("")
        
        interpretation.append("TOPIC INTERPRETABILITY:")
        interpretation.append("-" * 60)
        
        for topic_id, words in topics.items():
            top_words = ", ".join([word for word, _ in words[:5]])
            interpretation.append(f"Topic {topic_id}: {top_words}")
        
        interpretation.append("")
        interpretation.append("LIMITATIONS OF CLASSICAL NLP:")
        interpretation.append("-" * 60)
        interpretation.append("• Coherence scores measure statistical co-occurrence, not semantic meaning")
        interpretation.append("• Topics may contain unrelated words that frequently co-occur")
        interpretation.append("• No understanding of context, polysemy, or synonymy")
        interpretation.append("• Automatic theme labeling is unreliable")
        interpretation.append("• Results highly sensitive to preprocessing choices")
        interpretation.append("")
        
        return "\n".join(interpretation)
    
    def evaluate_clustering(self, labels: np.ndarray, 
                          feature_matrix: np.ndarray) -> Dict[str, float]:
        """
        Evaluate clustering results.
        
        Args:
            labels: Cluster labels
            feature_matrix: Document feature matrix
            
        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
        
        metrics = {}
        
        try:
            # Silhouette score (higher is better, range: -1 to 1)
            metrics['silhouette_score'] = silhouette_score(feature_matrix, labels)
            
            # Davies-Bouldin score (lower is better)
            metrics['davies_bouldin_score'] = davies_bouldin_score(feature_matrix, labels)
            
            # Calinski-Harabasz score (higher is better)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(feature_matrix, labels)
            
            self.logger.info(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
            self.logger.info(f"Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}")
            
        except Exception as e:
            self.logger.error(f"Clustering evaluation failed: {str(e)}")
        
        return metrics
    
    def compare_summaries(self, summaries: Dict[str, str]) -> Dict[str, Dict]:
        """
        Compare multiple summarization methods.
        
        Args:
            summaries: Dictionary mapping method names to summaries
            
        Returns:
            Comparison statistics
        """
        comparison = {}
        
        for method, summary in summaries.items():
            comparison[method] = {
                'length': len(summary),
                'word_count': len(summary.split()),
                'sentence_count': len([s for s in summary.split('.') if s.strip()])
            }
        
        return comparison
