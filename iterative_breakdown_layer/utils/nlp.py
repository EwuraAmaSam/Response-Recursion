"""
NLP utility functions.
"""

# Apply Python 3.12 compatibility patch before importing spacy
try:
    import sys
    import os
    # Add project root to path to import the fix
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        import fix_pydantic_python312
    except ImportError:
        pass  # Fix not available, continue anyway
except Exception:
    pass  # If fix fails, continue anyway

import spacy
from typing import Optional


_nlp_cache = {}


def get_nlp_model(model_name: str = "en_core_web_sm"):
    """
    Get or load a spaCy model (with caching).
    
    Args:
        model_name: spaCy model name
    
    Returns:
        Loaded spaCy model
    """
    if model_name not in _nlp_cache:
        try:
            _nlp_cache[model_name] = spacy.load(model_name)
        except OSError:
            # Fallback to small model
            if model_name != "en_core_web_sm":
                try:
                    _nlp_cache[model_name] = spacy.load("en_core_web_sm")
                except OSError:
                    raise RuntimeError(
                        f"Could not load spaCy model {model_name}. "
                        "Install with: python -m spacy download en_core_web_sm"
                    )
            else:
                raise RuntimeError(
                    f"Could not load spaCy model {model_name}. "
                    "Install with: python -m spacy download en_core_web_sm"
                )
    
    return _nlp_cache[model_name]


def clear_nlp_cache():
    """Clear the NLP model cache."""
    global _nlp_cache
    _nlp_cache.clear()

