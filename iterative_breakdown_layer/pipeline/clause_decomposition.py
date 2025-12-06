"""
Clause decomposition module using dependency parsing to break sentences
into independently evaluable clauses.
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
from typing import List, Dict, Any, Set
from collections import defaultdict


class ClauseDecomposer:
    """Decomposes sentences into clauses using dependency parsing."""
    
    # Relevant dependency types for clause extraction
    CLAUSE_DEPENDENCIES = {
        'ccomp',  # clausal complement
        'advcl',  # adverbial clause modifier
        'conj',   # conjunct
        'xcomp',  # open clausal complement
        'relcl',  # relative clause modifier
        'acl',    # clausal modifier of noun
        'appos'   # appositional modifier
    }
    
    def __init__(self, nlp_model=None):
        """
        Initialize the decomposer.
        
        Args:
            nlp_model: Pre-loaded spaCy model (optional)
        """
        if nlp_model is None:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise RuntimeError("spaCy model required for clause decomposition")
        else:
            self.nlp = nlp_model
    
    def extract_clauses(self, sentence: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract clauses from a sentence.
        
        Args:
            sentence: Sentence dict with 'text' and 'span' keys
        
        Returns:
            List of clause dictionaries
        """
        text = sentence['text']
        base_start = sentence.get('start_char', sentence['span'][0])
        
        doc = self.nlp(text)
        clauses = []
        
        # Extract main clause
        main_clause = self._extract_main_clause(doc, base_start)
        if main_clause:
            clauses.append(main_clause)
        
        # Extract subordinate clauses
        for token in doc:
            if token.dep_ in self.CLAUSE_DEPENDENCIES:
                clause = self._extract_subordinate_clause(
                    token, doc, base_start, sentence
                )
                if clause and clause not in clauses:
                    clauses.append(clause)
        
        # If no clauses found, return the sentence as a single clause
        if not clauses:
            clauses.append({
                "text": text,
                "span": [base_start, base_start + len(text)],
                "start_char": base_start,
                "end_char": base_start + len(text),
                "type": "main",
                "dependencies": []
            })
        
        return clauses
    
    def _extract_main_clause(self, doc, base_start: int) -> Dict[str, Any]:
        """Extract the main clause from a sentence."""
        root = [token for token in doc if token.dep_ == "ROOT"][0]
        
        # Get the subtree of the root
        main_clause_tokens = list(root.subtree)
        main_clause_text = doc[main_clause_tokens[0].i:main_clause_tokens[-1].i + 1].text
        
        start_char = base_start + main_clause_tokens[0].idx
        end_char = base_start + main_clause_tokens[-1].idx + len(main_clause_tokens[-1])
        
        return {
            "text": main_clause_text,
            "span": [start_char, end_char],
            "start_char": start_char,
            "end_char": end_char,
            "type": "main",
            "dependencies": []
        }
    
    def _extract_subordinate_clause(
        self, 
        token, 
        doc, 
        base_start: int, 
        sentence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract a subordinate clause starting from a token."""
        # Get the subtree of this clause
        clause_tokens = list(token.subtree)
        
        if not clause_tokens:
            return None
        
        clause_text = doc[clause_tokens[0].i:clause_tokens[-1].i + 1].text
        
        start_char = base_start + clause_tokens[0].idx
        end_char = base_start + clause_tokens[-1].idx + len(clause_tokens[-1])
        
        # Determine clause type from dependency
        clause_type_map = {
            'ccomp': 'complement',
            'advcl': 'adverbial',
            'conj': 'conjunctive',
            'xcomp': 'open_complement',
            'relcl': 'relative',
            'acl': 'adjectival',
            'appos': 'appositive'
        }
        
        return {
            "text": clause_text,
            "span": [start_char, end_char],
            "start_char": start_char,
            "end_char": end_char,
            "type": clause_type_map.get(token.dep_, 'subordinate'),
            "dependencies": [token.dep_],
            "head_token": token.head.text if token.head else None
        }
    
    def decompose_sentences(self, sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Decompose a list of sentences into clauses.
        
        Args:
            sentences: List of sentence dictionaries
        
        Returns:
            List of clause dictionaries
        """
        all_clauses = []
        
        for sentence in sentences:
            clauses = self.extract_clauses(sentence)
            all_clauses.extend(clauses)
        
        return all_clauses


def decompose_into_clauses(
    sentences: List[Dict[str, Any]], 
    nlp_model=None
) -> List[Dict[str, Any]]:
    """
    Convenience function for clause decomposition.
    
    Args:
        sentences: List of sentence dictionaries
        nlp_model: Optional pre-loaded spaCy model
    
    Returns:
        List of clause dictionaries
    """
    decomposer = ClauseDecomposer(nlp_model)
    return decomposer.decompose_sentences(sentences)

