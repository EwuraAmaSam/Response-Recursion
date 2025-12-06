"""
Utility functions for handling text spans and character offsets.
"""

from typing import List, Tuple, Dict, Any


def merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Merge overlapping or adjacent spans.
    
    Args:
        spans: List of (start, end) tuples
    
    Returns:
        Merged list of spans
    """
    if not spans:
        return []
    
    # Sort by start position
    sorted_spans = sorted(spans, key=lambda x: x[0])
    merged = [sorted_spans[0]]
    
    for current in sorted_spans[1:]:
        last = merged[-1]
        # If overlapping or adjacent, merge
        if current[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    
    return merged


def span_contains(span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
    """
    Check if span1 contains span2.
    
    Args:
        span1: Container span (start, end)
        span2: Contained span (start, end)
    
    Returns:
        True if span1 contains span2
    """
    return span1[0] <= span2[0] and span1[1] >= span2[1]


def span_overlaps(span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
    """
    Check if two spans overlap.
    
    Args:
        span1: First span (start, end)
        span2: Second span (start, end)
    
    Returns:
        True if spans overlap
    """
    return not (span1[1] <= span2[0] or span2[1] <= span1[0])


def get_text_from_span(text: str, span: Tuple[int, int]) -> str:
    """
    Extract text from a span.
    
    Args:
        text: Full text
        span: (start, end) tuple
    
    Returns:
        Extracted text
    """
    start, end = span
    return text[start:end]


def adjust_spans(
    spans: List[Dict[str, Any]], 
    offset: int
) -> List[Dict[str, Any]]:
    """
    Adjust span positions by an offset.
    
    Args:
        spans: List of span dictionaries with 'span' key
        offset: Offset to add to all spans
    
    Returns:
        Adjusted spans
    """
    adjusted = []
    for span_dict in spans:
        adjusted_dict = span_dict.copy()
        if 'span' in adjusted_dict:
            adjusted_dict['span'] = [
                adjusted_dict['span'][0] + offset,
                adjusted_dict['span'][1] + offset
            ]
        if 'start_char' in adjusted_dict:
            adjusted_dict['start_char'] += offset
        if 'end_char' in adjusted_dict:
            adjusted_dict['end_char'] += offset
        adjusted.append(adjusted_dict)
    
    return adjusted



