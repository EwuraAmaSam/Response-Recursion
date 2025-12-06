"""
Preprocessing module for the Iterative Breakdown Layer.
Handles text normalization, whitespace cleanup, and preservation of code/quotations.
"""

import re
from typing import Dict, Any, Tuple


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving structure."""
    # Replace multiple spaces with single space, but preserve newlines
    text = re.sub(r'[ \t]+', ' ', text)
    # Normalize newlines
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\r', '\n', text)
    # Remove trailing whitespace from lines
    text = '\n'.join(line.rstrip() for line in text.split('\n'))
    return text.strip()


def preserve_code_fences(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Extract and preserve code fences and quotations.
    Returns (text_with_placeholders, mapping_dict)
    """
    placeholders = {}
    placeholder_counter = 0
    
    # Match code fences (```...``` or ```language...```)
    code_pattern = r'```[\s\S]*?```'
    for match in re.finditer(code_pattern, text):
        placeholder = f"__CODE_FENCE_{placeholder_counter}__"
        placeholders[placeholder] = match.group(0)
        text = text.replace(match.group(0), placeholder, 1)
        placeholder_counter += 1
    
    # Match inline code (`...`)
    inline_code_pattern = r'`[^`]+`'
    for match in re.finditer(inline_code_pattern, text):
        placeholder = f"__INLINE_CODE_{placeholder_counter}__"
        placeholders[placeholder] = match.group(0)
        text = text.replace(match.group(0), placeholder, 1)
        placeholder_counter += 1
    
    # Match quotations (preserve common quote patterns)
    quote_pattern = r'"[^"]*"'
    for match in re.finditer(quote_pattern, text):
        placeholder = f"__QUOTE_{placeholder_counter}__"
        placeholders[placeholder] = match.group(0)
        text = text.replace(match.group(0), placeholder, 1)
        placeholder_counter += 1
    
    return text, placeholders


def restore_code_fences(text: str, placeholders: Dict[str, str]) -> str:
    """Restore code fences and quotations from placeholders."""
    for placeholder, original in placeholders.items():
        text = text.replace(placeholder, original)
    return text


def remove_boilerplate(text: str) -> str:
    """
    Remove clearly redundant boilerplate only.
    Conservative approach - only remove obvious patterns.
    """
    # Common LLM boilerplate patterns
    boilerplate_patterns = [
        r'^(As an AI|I am an AI|I\'m an AI),?\s*',
        r'^(I can|I cannot|I\'m unable to),?\s*',
        r'^(Let me|Allow me to),?\s*',
    ]
    
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    return text.strip()


def preprocess_text(text: str, preserve_structure: bool = True) -> Dict[str, Any]:
    """
    Main preprocessing function.
    
    Args:
        text: Raw input text
        preserve_structure: Whether to preserve code fences and quotations
    
    Returns:
        Dictionary with preprocessed text and metadata
    """
    if not text:
        return {
            "preprocessed_text": "",
            "placeholders": {},
            "metadata": {}
        }
    
    original_length = len(text)
    
    # Preserve code fences and quotations if requested
    if preserve_structure:
        text, placeholders = preserve_code_fences(text)
    else:
        placeholders = {}
    
    # Normalize whitespace
    text = normalize_whitespace(text)
    
    # Remove boilerplate (conservative)
    text = remove_boilerplate(text)
    
    # Final normalization
    text = normalize_whitespace(text)
    
    return {
        "preprocessed_text": text,
        "placeholders": placeholders,
        "metadata": {
            "original_length": original_length,
            "preprocessed_length": len(text),
            "preserved_structure": preserve_structure
        }
    }

