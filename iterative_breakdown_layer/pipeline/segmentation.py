"""
Sentence segmentation module using spaCy for sentence boundary detection.
"""

# Apply Python 3.12 compatibility patch before importing spacy
try:
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        import fix_pydantic_python312
    except ImportError:
        pass
except Exception:
    pass

import spacy
from typing import List, Dict, Any, Tuple
import warnings


class SentenceSegmenter:
    """Handles sentence segmentation using spaCy."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the segmenter.
        
        Args:
            model_name: spaCy model name (default: en_core_web_sm, 
                       recommended: en_core_web_trf)
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            warnings.warn(
                f"Model {model_name} not found. Falling back to en_core_web_sm. "
                f"Install with: python -m spacy download {model_name}"
            )
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise RuntimeError(
                    "No spaCy model found. Please install one with: "
                    "python -m spacy download en_core_web_sm"
                )
    
    def segment(self, text: str) -> List[Dict[str, Any]]:
        """
        Segment text into sentences with spans.
        
        Args:
            text: Preprocessed text
        
        Returns:
            List of sentence dictionaries with text and character spans
        """
        if not text.strip():
            return []
        
        doc = self.nlp(text)
        sentences = []
        
        for sent in doc.sents:
            start_char = sent.start_char
            end_char = sent.end_char
            sentence_text = sent.text.strip()
            
            if sentence_text:  # Only include non-empty sentences
                sentences.append({
                    "text": sentence_text,
                    "span": [start_char, end_char],
                    "start_char": start_char,
                    "end_char": end_char
                })
        
        return sentences
    
    def segment_with_metadata(self, text: str) -> Dict[str, Any]:
        """
        Segment text and return with metadata.
        
        Args:
            text: Preprocessed text
        
        Returns:
            Dictionary with sentences and metadata
        """
        sentences = self.segment(text)
        
        return {
            "sentences": sentences,
            "metadata": {
                "total_sentences": len(sentences),
                "text_length": len(text)
            }
        }


def segment_text(text: str, model_name: str = "en_core_web_sm") -> List[Dict[str, Any]]:
    """
    Convenience function for sentence segmentation.
    
    Args:
        text: Preprocessed text
        model_name: spaCy model name
    
    Returns:
        List of sentence dictionaries
    """
    segmenter = SentenceSegmenter(model_name)
    return segmenter.segment(text)

