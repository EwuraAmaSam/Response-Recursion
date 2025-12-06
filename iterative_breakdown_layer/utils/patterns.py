"""
Pattern matching utilities for text analysis.
"""

import re
from typing import List, Dict, Any, Pattern


def compile_patterns(patterns: List[str], flags: int = re.IGNORECASE) -> List[Pattern]:
    """
    Compile a list of regex patterns.
    
    Args:
        patterns: List of pattern strings
        flags: Regex flags
    
    Returns:
        List of compiled patterns
    """
    return [re.compile(pattern, flags) for pattern in patterns]


def find_all_matches(text: str, pattern: Pattern) -> List[Dict[str, Any]]:
    """
    Find all matches of a pattern in text.
    
    Args:
        text: Text to search
        pattern: Compiled regex pattern
    
    Returns:
        List of match dictionaries with 'text', 'span', 'start', 'end'
    """
    matches = []
    for match in pattern.finditer(text):
        matches.append({
            'text': match.group(0),
            'span': [match.start(), match.end()],
            'start': match.start(),
            'end': match.end(),
            'groups': match.groups()
        })
    return matches


def match_any_pattern(text: str, patterns: List[Pattern]) -> bool:
    """
    Check if text matches any of the patterns.
    
    Args:
        text: Text to check
        patterns: List of compiled patterns
    
    Returns:
        True if any pattern matches
    """
    for pattern in patterns:
        if pattern.search(text):
            return True
    return False


def extract_pattern_groups(text: str, pattern: Pattern) -> List[Dict[str, Any]]:
    """
    Extract pattern matches with named groups.
    
    Args:
        text: Text to search
        pattern: Compiled regex pattern with named groups
    
    Returns:
        List of dictionaries with group names as keys
    """
    matches = []
    for match in pattern.finditer(text):
        group_dict = match.groupdict()
        group_dict['full_match'] = match.group(0)
        group_dict['span'] = [match.start(), match.end()]
        matches.append(group_dict)
    return matches



