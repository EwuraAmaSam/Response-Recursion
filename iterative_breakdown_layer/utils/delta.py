"""
Delta computation utilities for comparing iterations.
"""

from typing import List, Dict, Any, Set
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class DeltaComputer:
    """Computes deltas between iterations."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize delta computer.
        
        Args:
            similarity_threshold: Threshold for considering statements similar
        """
        self.similarity_threshold = similarity_threshold
        self.embedding_model = None
    
    def _get_embedding_model(self):
        """Lazy load embedding model."""
        if self.embedding_model is None:
            try:
                self.embedding_model = SentenceTransformer("all-mpnet-base-v2")
            except Exception:
                try:
                    self.embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
                except Exception:
                    raise RuntimeError("Could not load embedding model for delta computation")
        return self.embedding_model
    
    def compute_semantic_similarity(
        self, 
        text1: str, 
        text2: str
    ) -> float:
        """
        Compute semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Similarity score (0-1)
        """
        model = self._get_embedding_model()
        embeddings = model.encode([text1, text2], convert_to_numpy=True)
        similarity = cosine_similarity(embeddings[0:1], embeddings[1:2])[0][0]
        return float(similarity)
    
    def find_similar_statements(
        self,
        current_statements: List[Dict[str, Any]],
        prior_statements: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Find semantically similar statements between iterations.
        
        Args:
            current_statements: Current iteration statements
            prior_statements: Prior iteration statements
        
        Returns:
            Dictionary mapping current statement IDs to prior statement IDs
        """
        similarity_map = {}
        
        for current_stmt in current_statements:
            current_text = current_stmt.get('text', '')
            current_id = current_stmt.get('id', '')
            best_match_id = None
            best_similarity = 0.0
            
            for prior_stmt in prior_statements:
                prior_text = prior_stmt.get('text', '')
                prior_id = prior_stmt.get('id', '')
                
                similarity = self.compute_semantic_similarity(current_text, prior_text)
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match_id = prior_id
            
            if best_match_id:
                similarity_map[current_id] = best_match_id
        
        return similarity_map
    
    def compute_detailed_delta(
        self,
        current_statements: List[Dict[str, Any]],
        prior_statements: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute detailed delta with semantic matching.
        
        Args:
            current_statements: Current iteration statements
            prior_statements: Prior iteration statements
        
        Returns:
            Detailed delta dictionary
        """
        # Find similar statements
        similarity_map = self.find_similar_statements(current_statements, prior_statements)
        
        # Create ID maps
        current_by_id = {stmt.get('id'): stmt for stmt in current_statements}
        prior_by_id = {stmt.get('id'): stmt for stmt in prior_statements}
        
        # Find new statements (not similar to any prior)
        new_ids = [
            stmt_id for stmt_id in current_by_id.keys()
            if stmt_id not in similarity_map.values()
        ]
        
        # Find removed statements (prior statements not matched)
        matched_prior_ids = set(similarity_map.values())
        removed_ids = [
            stmt_id for stmt_id in prior_by_id.keys()
            if stmt_id not in matched_prior_ids
        ]
        
        # Find changed statements (similar but text/type changed)
        changed_ids = []
        for current_id, prior_id in similarity_map.items():
            current_stmt = current_by_id[current_id]
            prior_stmt = prior_by_id[prior_id]
            
            # Check if text or type changed
            if (current_stmt.get('text') != prior_stmt.get('text') or
                current_stmt.get('type') != prior_stmt.get('type')):
                changed_ids.append({
                    'current_id': current_id,
                    'prior_id': prior_id,
                    'changes': {
                        'text_changed': current_stmt.get('text') != prior_stmt.get('text'),
                        'type_changed': current_stmt.get('type') != prior_stmt.get('type')
                    }
                })
        
        return {
            'new': new_ids,
            'removed': removed_ids,
            'changed': changed_ids,
            'similarity_map': similarity_map,
            'total_current': len(current_statements),
            'total_prior': len(prior_statements)
        }


def compute_delta(
    current_statements: List[Dict[str, Any]],
    prior_statements: List[Dict[str, Any]],
    use_semantic: bool = False
) -> Dict[str, Any]:
    """
    Convenience function for delta computation.
    
    Args:
        current_statements: Current iteration statements
        prior_statements: Prior iteration statements
        use_semantic: Whether to use semantic similarity matching
    
    Returns:
        Delta dictionary
    """
    if use_semantic:
        computer = DeltaComputer()
        return computer.compute_detailed_delta(current_statements, prior_statements)
    else:
        # Simple ID-based comparison
        current_ids = {stmt.get('id') for stmt in current_statements}
        prior_ids = {stmt.get('id') for stmt in prior_statements}
        
        return {
            'new': list(current_ids - prior_ids),
            'removed': list(prior_ids - current_ids),
            'changed': [],
            'total_current': len(current_statements),
            'total_prior': len(prior_statements)
        }



