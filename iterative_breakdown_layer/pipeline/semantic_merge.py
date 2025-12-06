"""
Semantic merge/split module that recombines fragments and splits
multi-proposition clauses using semantic similarity.
"""

from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticMerger:
    """Handles semantic merging and splitting of clauses."""
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initialize the semantic merger.
        
        Args:
            model_name: Hugging Face sentence transformer model name
        """
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            # Fallback to a smaller model if available
            try:
                self.model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
            except Exception:
                raise RuntimeError(
                    f"Could not load sentence transformer model {model_name}. "
                    "Please install: pip install sentence-transformers"
                )
        self.similarity_threshold = 0.75  # Threshold for merging
    
    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for a list of texts."""
        if not texts:
            return np.array([])
        return self.model.encode(texts, convert_to_numpy=True)
    
    def merge_similar_clauses(self, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge clauses that represent the same concept.
        
        Args:
            clauses: List of clause dictionaries
        
        Returns:
            Merged list of clauses
        """
        if len(clauses) <= 1:
            return clauses
        
        # Extract texts for embedding
        texts = [clause['text'] for clause in clauses]
        embeddings = self.compute_embeddings(texts)
        
        # Find similar clauses
        merged_indices = set()
        merged_clauses = []
        
        for i, clause in enumerate(clauses):
            if i in merged_indices:
                continue
            
            # Find similar clauses
            similar_indices = [i]
            for j in range(i + 1, len(clauses)):
                if j in merged_indices:
                    continue
                
                similarity = cosine_similarity(
                    embeddings[i:i+1], 
                    embeddings[j:j+1]
                )[0][0]
                
                if similarity >= self.similarity_threshold:
                    similar_indices.append(j)
                    merged_indices.add(j)
            
            # Merge similar clauses
            if len(similar_indices) > 1:
                merged_text = " ".join([clauses[idx]['text'] for idx in similar_indices])
                # Use span from first to last clause
                start_span = min(clauses[idx]['span'][0] for idx in similar_indices)
                end_span = max(clauses[idx]['span'][1] for idx in similar_indices)
                
                merged_clause = {
                    "text": merged_text,
                    "span": [start_span, end_span],
                    "start_char": start_span,
                    "end_char": end_span,
                    "type": clauses[i].get("type", "merged"),
                    "dependencies": clauses[i].get("dependencies", []),
                    "merged_from": [clauses[idx].get("id", f"clause_{idx}") for idx in similar_indices]
                }
                merged_clauses.append(merged_clause)
            else:
                merged_clauses.append(clause)
        
        return merged_clauses
    
    def split_multi_proposition_clauses(
        self, 
        clauses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Split clauses that contain multiple testable propositions.
        Uses heuristics and semantic analysis.
        
        Args:
            clauses: List of clause dictionaries
        
        Returns:
            Split list of clauses
        """
        split_clauses = []
        
        for clause in clauses:
            text = clause['text']
            
            # Heuristic: Check for conjunction patterns
            if self._has_multiple_propositions(text):
                sub_clauses = self._split_by_conjunctions(clause)
                split_clauses.extend(sub_clauses)
            else:
                split_clauses.append(clause)
        
        return split_clauses
    
    def _has_multiple_propositions(self, text: str) -> bool:
        """Heuristic check for multiple propositions."""
        # Check for coordinating conjunctions
        conjunctions = [' and ', ' but ', ' or ', ' yet ', ' so ', ' nor ']
        for conj in conjunctions:
            if conj in text.lower():
                return True
        
        # Check for semicolons
        if ';' in text:
            return True
        
        return False
    
    def _split_by_conjunctions(self, clause: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a clause by conjunctions."""
        text = clause['text']
        base_start = clause['span'][0]
        
        # Simple splitting by common conjunctions
        split_patterns = [
            r'\s+and\s+',
            r'\s+but\s+',
            r'\s+or\s+',
            r'\s+yet\s+',
            r'\s+so\s+',
            r'\s+nor\s+',
            r';\s+'
        ]
        
        import re
        parts = [text]
        for pattern in split_patterns:
            new_parts = []
            for part in parts:
                splits = re.split(pattern, part, flags=re.IGNORECASE)
                new_parts.extend(splits)
            parts = new_parts
        
        # Filter out empty parts
        parts = [p.strip() for p in parts if p.strip()]
        
        if len(parts) <= 1:
            return [clause]
        
        # Create clause objects for each part
        split_clauses = []
        current_pos = 0
        
        for i, part in enumerate(parts):
            # Approximate span (this is a simplification)
            start_char = base_start + text.find(part, current_pos)
            end_char = start_char + len(part)
            current_pos = text.find(part, current_pos) + len(part)
            
            split_clause = {
                "text": part,
                "span": [start_char, end_char],
                "start_char": start_char,
                "end_char": end_char,
                "type": clause.get("type", "split"),
                "dependencies": clause.get("dependencies", []),
                "split_from": clause.get("id", "original")
            }
            split_clauses.append(split_clause)
        
        return split_clauses
    
    def process_clauses(self, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main processing function: split then merge.
        
        Args:
            clauses: List of clause dictionaries
        
        Returns:
            Processed list of Core Statements
        """
        # First, split multi-proposition clauses
        split_clauses = self.split_multi_proposition_clauses(clauses)
        
        # Then, merge similar clauses
        merged_clauses = self.merge_similar_clauses(split_clauses)
        
        # Assign IDs
        for i, clause in enumerate(merged_clauses):
            clause['id'] = f"CS_{i:03d}"
        
        return merged_clauses


def process_semantic_merge(
    clauses: List[Dict[str, Any]], 
    model_name: str = "all-mpnet-base-v2"
) -> List[Dict[str, Any]]:
    """
    Convenience function for semantic merge/split.
    
    Args:
        clauses: List of clause dictionaries
        model_name: Sentence transformer model name
    
    Returns:
        Processed list of Core Statements
    """
    merger = SemanticMerger(model_name)
    return merger.process_clauses(clauses)



