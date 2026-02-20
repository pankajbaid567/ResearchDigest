"""
Document Loader Module
Handles loading and extracting text from various document formats (.txt, .pdf).
"""

import os
from typing import List, Dict, Optional, Union
import logging
from pathlib import Path

try:
    import PyPDF2
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logging.warning("PDF libraries not available. Install PyPDF2 and pdfplumber for PDF support.")

from utils import clean_text, validate_file_path


class DocumentLoader:
    """
    Load and extract text from documents.
    Supports: .txt, .md, .pdf
    """
    
    SUPPORTED_TEXT_EXTENSIONS = ['.txt', '.md', '.text']
    SUPPORTED_PDF_EXTENSIONS = ['.pdf']
    
    def __init__(self):
        """Initialize document loader."""
        self.logger = logging.getLogger(__name__)
    
    def load_single_document(self, file_path: str) -> Optional[str]:
        """
        Load a single document and extract text.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Extracted text or None if failed
        """
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            return None
        
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext in self.SUPPORTED_TEXT_EXTENSIONS:
                return self._load_text_file(file_path)
            elif file_ext in self.SUPPORTED_PDF_EXTENSIONS:
                return self._load_pdf_file(file_path)
            else:
                self.logger.error(f"Unsupported file format: {file_ext}")
                return None
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {str(e)}")
            return None
    
    def load_multiple_documents(self, file_paths: List[str]) -> Dict[str, str]:
        """
        Load multiple documents.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Dictionary mapping file paths to extracted text
        """
        documents = {}
        
        for file_path in file_paths:
            text = self.load_single_document(file_path)
            if text:
                documents[file_path] = text
                self.logger.info(f"Loaded: {file_path} ({len(text)} characters)")
            else:
                self.logger.warning(f"Failed to load: {file_path}")
        
        return documents
    
    def _load_text_file(self, file_path: str) -> str:
        """
        Load plain text file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            File contents as string
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        return clean_text(text)
    
    def _load_pdf_file(self, file_path: str) -> str:
        """
        Extract text from PDF file using pdfplumber (primary) or PyPDF2 (fallback).
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        if not PDF_SUPPORT:
            raise ImportError("PDF support not available. Install PyPDF2 and pdfplumber.")
        
        text = ""
        
        # Try pdfplumber first (better text extraction)
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            if text.strip():
                return clean_text(text)
        except Exception as e:
            self.logger.warning(f"pdfplumber failed for {file_path}: {str(e)}. Trying PyPDF2...")
        
        # Fallback to PyPDF2
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            return clean_text(text)
        except Exception as e:
            self.logger.error(f"PyPDF2 also failed for {file_path}: {str(e)}")
            raise
    
    def load_from_uploaded_files(self, uploaded_files) -> Dict[str, str]:
        """
        Load documents from Streamlit uploaded files.
        
        Args:
            uploaded_files: Streamlit UploadedFile objects
            
        Returns:
            Dictionary mapping filenames to extracted text
        """
        documents = {}
        
        for uploaded_file in uploaded_files:
            try:
                file_ext = Path(uploaded_file.name).suffix.lower()
                
                if file_ext in self.SUPPORTED_TEXT_EXTENSIONS:
                    # Read text file
                    text = uploaded_file.read().decode('utf-8', errors='ignore')
                    documents[uploaded_file.name] = clean_text(text)
                    
                elif file_ext in self.SUPPORTED_PDF_EXTENSIONS and PDF_SUPPORT:
                    # Save temporarily to extract PDF text
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                    
                    try:
                        text = self._load_pdf_file(tmp_path)
                        documents[uploaded_file.name] = text
                    finally:
                        os.unlink(tmp_path)
                else:
                    self.logger.warning(f"Unsupported file type: {uploaded_file.name}")
                    
            except Exception as e:
                self.logger.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        return documents
    
    @staticmethod
    def get_supported_extensions() -> List[str]:
        """
        Get list of all supported file extensions.
        
        Returns:
            List of file extensions
        """
        extensions = DocumentLoader.SUPPORTED_TEXT_EXTENSIONS.copy()
        if PDF_SUPPORT:
            extensions.extend(DocumentLoader.SUPPORTED_PDF_EXTENSIONS)
        return extensions
