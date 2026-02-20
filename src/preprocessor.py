"""
Text Preprocessing Module
Handles tokenization, stop-word removal, and lemmatization using NLTK and spaCy.
"""

import re
from typing import List, Dict, Any, Optional
import logging

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. Using NLTK only.")


class TextPreprocessor:
    """
    Preprocess text using classical NLP techniques.
    Handles: tokenization, stop-word removal, lemmatization.
    """
    
    def __init__(self, use_spacy: bool = True, language: str = 'english'):
        """
        Initialize preprocessor.
        
        Args:
            use_spacy: Use spaCy for lemmatization if available
            language: Language for stop words and processing
        """
        self.logger = logging.getLogger(__name__)
        self.language = language
        
        # Download required NLTK data
        self._download_nltk_resources()
        
        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words(language))
        except Exception:
            self.logger.warning(f"Stopwords for '{language}' not available. Using default.")
            self.stop_words = set()
        
        # Add custom academic stop words
        self.stop_words.update([
            'et', 'al', 'fig', 'figure', 'table', 'section', 'chapter',
            'paper', 'study', 'research', 'pp', 'vol', 'doi', 'isbn'
        ])
        
        # Initialize spaCy if available and requested
        self.nlp = None
        if use_spacy and SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load('en_core_web_sm')
                self.logger.info("Loaded spaCy model: en_core_web_sm")
            except Exception as e:
                self.logger.warning(f"Could not load spaCy model: {str(e)}. Using NLTK.")
                self.nlp = None
    
    def _download_nltk_resources(self):
        """Download required NLTK resources."""
        resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except Exception as e:
                    self.logger.warning(f"Could not download NLTK resource '{resource}': {str(e)}")
    
    def preprocess_text(self, text: str, remove_stopwords: bool = True, 
                       lemmatize: bool = True) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline.
        
        Args:
            text: Input text to preprocess
            remove_stopwords: Whether to remove stop words
            lemmatize: Whether to lemmatize tokens
            
        Returns:
            Dictionary containing:
                - original_text: Original input
                - cleaned_text: Cleaned version
                - sentences: List of sentences
                - tokens: Word tokens
                - tokens_no_stopwords: Tokens with stop words removed
                - lemmatized_tokens: Lemmatized tokens
        """
        # Clean text
        cleaned = self._clean_text(text)
        
        # Sentence tokenization
        sentences = self.tokenize_sentences(cleaned)
        
        # Word tokenization
        tokens = self.tokenize_words(cleaned)
        
        # Remove stop words
        tokens_no_stopwords = tokens
        if remove_stopwords:
            tokens_no_stopwords = self.remove_stopwords(tokens)
        
        # Lemmatization
        lemmatized = tokens_no_stopwords
        if lemmatize:
            if self.nlp:
                lemmatized = self.lemmatize_spacy(tokens_no_stopwords)
            else:
                lemmatized = self.lemmatize_nltk(tokens_no_stopwords)
        
        return {
            'original_text': text,
            'cleaned_text': cleaned,
            'sentences': sentences,
            'tokens': tokens,
            'tokens_no_stopwords': tokens_no_stopwords,
            'lemmatized_tokens': lemmatized,
            'token_count': len(tokens),
            'sentence_count': len(sentences)
        }
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters and normalizing whitespace.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?]', ' ', text)
        
        # Remove numbers (optional - comment out if you want to keep numbers)
        # text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Tokenize text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        try:
            sentences = sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            self.logger.error(f"Sentence tokenization failed: {str(e)}")
            # Fallback: split by periods
            return [s.strip() + '.' for s in text.split('.') if s.strip()]
    
    def tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of word tokens
        """
        try:
            tokens = word_tokenize(text)
            # Filter out non-alphabetic tokens and single characters
            tokens = [t for t in tokens if t.isalpha() and len(t) > 1]
            return tokens
        except Exception as e:
            self.logger.error(f"Word tokenization failed: {str(e)}")
            # Fallback: simple split
            return [w for w in text.split() if w.isalpha() and len(w) > 1]
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stop words from token list.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            Filtered token list
        """
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize_nltk(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens using NLTK WordNetLemmatizer.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def lemmatize_spacy(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens using spaCy.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            Lemmatized tokens
        """
        if not self.nlp:
            return self.lemmatize_nltk(tokens)
        
        # Process tokens as a document
        text = ' '.join(tokens)
        doc = self.nlp(text)
        
        # Extract lemmas
        lemmas = [token.lemma_ for token in doc if token.is_alpha]
        return lemmas
    
    def preprocess_documents(self, documents: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """
        Preprocess multiple documents.
        
        Args:
            documents: Dictionary mapping document IDs to text
            
        Returns:
            Dictionary mapping document IDs to preprocessed data
        """
        preprocessed = {}
        
        for doc_id, text in documents.items():
            self.logger.info(f"Preprocessing: {doc_id}")
            preprocessed[doc_id] = self.preprocess_text(text)
        
        return preprocessed
