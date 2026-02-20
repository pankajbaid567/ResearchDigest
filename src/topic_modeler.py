"""
Topic Modeling Module
Implements LDA (Latent Dirichlet Allocation) using Gensim for topic extraction.
"""

from typing import List, Dict, Tuple, Optional
import logging
import numpy as np

from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel


class TopicModeler:
    """
    Perform topic modeling using LDA (Gensim).
    Extracts latent topics from document collections.
    """
    
    def __init__(self):
        """Initialize topic modeler."""
        self.logger = logging.getLogger(__name__)
        self.lda_model = None
        self.dictionary = None
        self.corpus = None
        self.texts = None
    
    def prepare_corpus(self, documents: List[List[str]]) -> Tuple[corpora.Dictionary, List]:
        """
        Prepare corpus and dictionary for LDA.
        
        Args:
            documents: List of tokenized documents (each doc is a list of tokens)
            
        Returns:
            Tuple of (dictionary, corpus)
        """
        # Validate input
        if not documents:
            raise ValueError("No documents provided for topic modeling.")
        
        # Flatten and check total tokens
        total_tokens = sum(len(doc) for doc in documents)
        if total_tokens == 0:
            raise ValueError("All documents are empty after preprocessing. Please provide more text content.")
        
        # Create dictionary
        self.dictionary = corpora.Dictionary(documents)
        
        num_docs = len(documents)
        
        # Adapt filter_extremes based on document count
        if num_docs == 1:
            # Single document: don't filter at all
            self.logger.info("Single document mode: skipping dictionary filtering")
            # No filtering - keep all terms
        elif num_docs <= 3:
            # Very small collection: minimal filtering
            self.dictionary.filter_extremes(
                no_below=1,  # Appear in at least 1 doc
                no_above=1.0,  # No upper limit
                keep_n=500
            )
        else:
            # Normal collection: standard filtering
            self.dictionary.filter_extremes(
                no_below=min(2, num_docs // 2),  # At least 2, but not more than half
                no_above=0.8,
                keep_n=1000
            )
        
        # Validate dictionary after filtering
        if len(self.dictionary) == 0:
            raise ValueError(
                f"No terms remaining after dictionary filtering. "
                f"Your text may be too short or use only common stop words. "
                f"Documents: {num_docs}, Total tokens before filtering: {total_tokens}. "
                f"Please provide more substantial content."
            )
        
        # Create corpus (bag-of-words representation)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in documents]
        
        # Validate corpus - check if any documents have content
        non_empty_docs = sum(1 for doc in self.corpus if len(doc) > 0)
        if non_empty_docs == 0:
            raise ValueError(
                "All documents became empty after dictionary processing. "
                "This usually means the text is too short or contains only filtered terms."
            )
        
        self.texts = documents
        
        self.logger.info(f"Dictionary size: {len(self.dictionary)}")
        self.logger.info(f"Corpus size: {len(self.corpus)}, Non-empty docs: {non_empty_docs}")
        
        return self.dictionary, self.corpus
    
    def train_lda_model(self, documents: List[List[str]], num_topics: int = 5,
                       passes: int = 15, iterations: int = 100,
                       alpha: str = 'auto', eta: str = 'auto',
                       random_state: int = 42) -> LdaModel:
        """
        Train LDA model on documents.
        
        Args:
            documents: List of tokenized documents
            num_topics: Number of topics to extract
            passes: Number of passes through corpus
            iterations: Maximum iterations for convergence
            alpha: Document-topic density (prior)
            eta: Topic-word density (prior)
            random_state: Random seed for reproducibility
            
        Returns:
            Trained LDA model
        """
        # Prepare corpus
        self.prepare_corpus(documents)
        
        self.logger.info(f"Training LDA with {num_topics} topics...")
        
        # Train LDA model
        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=num_topics,
            passes=passes,
            iterations=iterations,
            alpha=alpha,
            eta=eta,
            random_state=random_state,
            per_word_topics=True
        )
        
        self.logger.info("LDA training complete")
        
        return self.lda_model
    
    def get_topics(self, num_words: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """
        Extract topics with top words and their weights.
        
        Args:
            num_words: Number of words to include per topic
            
        Returns:
            Dictionary mapping topic_id to list of (word, weight) tuples
        """
        if self.lda_model is None:
            raise ValueError("Model not trained. Call train_lda_model first.")
        
        topics = {}
        
        for topic_id in range(self.lda_model.num_topics):
            # Get top words for this topic
            topic_words = self.lda_model.show_topic(topic_id, topn=num_words)
            topics[topic_id] = topic_words
        
        return topics
    
    def get_document_topics(self, document_corpus: Optional[List] = None) -> List[List[Tuple[int, float]]]:
        """
        Get topic distribution for each document.
        
        Args:
            document_corpus: Corpus to analyze (uses training corpus if None)
            
        Returns:
            List of topic distributions for each document
        """
        if self.lda_model is None:
            raise ValueError("Model not trained.")
        
        corpus_to_use = document_corpus if document_corpus is not None else self.corpus
        
        doc_topics = []
        for doc in corpus_to_use:
            topics = self.lda_model.get_document_topics(doc)
            doc_topics.append(topics)
        
        return doc_topics
    
    def assign_dominant_topic(self, doc_topics: List[List[Tuple[int, float]]]) -> List[Dict]:
        """
        Assign dominant topic to each document.
        
        Args:
            doc_topics: Topic distributions for documents
            
        Returns:
            List of dictionaries with dominant topic info
        """
        dominant_topics = []
        
        for doc_idx, topics in enumerate(doc_topics):
            if not topics:
                dominant_topics.append({
                    'doc_idx': doc_idx,
                    'dominant_topic': -1,
                    'topic_weight': 0.0,
                    'all_topics': []
                })
                continue
            
            # Sort by weight
            sorted_topics = sorted(topics, key=lambda x: x[1], reverse=True)
            dominant = sorted_topics[0]
            
            dominant_topics.append({
                'doc_idx': doc_idx,
                'dominant_topic': dominant[0],
                'topic_weight': dominant[1],
                'all_topics': sorted_topics
            })
        
        return dominant_topics
    
    def calculate_coherence(self, coherence_type: str = 'c_v') -> float:
        """
        Calculate topic coherence score.
        
        Args:
            coherence_type: Type of coherence ('c_v', 'u_mass', 'c_uci', 'c_npmi')
            
        Returns:
            Coherence score
        """
        if self.lda_model is None or self.texts is None:
            raise ValueError("Model not trained or texts not available.")
        
        coherence_model = CoherenceModel(
            model=self.lda_model,
            texts=self.texts,
            dictionary=self.dictionary,
            coherence=coherence_type
        )
        
        coherence_score = coherence_model.get_coherence()
        
        self.logger.info(f"Coherence score ({coherence_type}): {coherence_score:.4f}")
        
        return coherence_score
    
    def get_topic_keywords(self, num_keywords: int = 10) -> Dict[int, List[str]]:
        """
        Get just the keywords (no weights) for each topic.
        
        Args:
            num_keywords: Number of keywords per topic
            
        Returns:
            Dictionary mapping topic_id to list of keywords
        """
        topics = self.get_topics(num_words=num_keywords)
        
        keywords = {}
        for topic_id, words_weights in topics.items():
            keywords[topic_id] = [word for word, weight in words_weights]
        
        return keywords
    
    def print_topics(self, num_words: int = 10) -> None:
        """
        Print topics in readable format.
        
        Args:
            num_words: Number of words per topic
        """
        if self.lda_model is None:
            raise ValueError("Model not trained.")
        
        self.logger.info("\n" + "="*60)
        self.logger.info("DISCOVERED TOPICS")
        self.logger.info("="*60)
        
        topics = self.get_topics(num_words)
        
        for topic_id, words in topics.items():
            words_str = ", ".join([f"{word}({weight:.3f})" for word, weight in words])
            self.logger.info(f"\nTopic {topic_id}: {words_str}")
        
        self.logger.info("="*60 + "\n")
