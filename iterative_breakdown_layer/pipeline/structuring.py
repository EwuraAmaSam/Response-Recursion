"""
Structuring module that packages outputs into embedding and prompt-generation profiles.
"""

from typing import List, Dict, Any, Optional
import json
from datetime import datetime


class Structurer:
    """Structures outputs into final profiles."""
    
    def __init__(self):
        """Initialize the structurer."""
        pass
    
    def create_embedding_units(
        self, 
        statements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Create embedding units from statements.
        
        Args:
            statements: List of Core Statement dictionaries
        
        Returns:
            List of embedding unit dictionaries
        """
        embedding_units = []
        
        for statement in statements:
            text = statement.get('text', '')
            cs_id = statement.get('id', '')
            
            # Clean text for embedding (lowercase, normalize)
            clean_text = text.lower().strip()
            # Remove extra whitespace
            clean_text = ' '.join(clean_text.split())
            
            embedding_unit = {
                'cs_id': cs_id,
                'clean_text': clean_text,
                'aggregation_hint': 'sentence',  # Could be 'clause', 'sentence', 'paragraph'
                'span': statement.get('span', []),
                'type': statement.get('type', 'Unknown')
            }
            embedding_units.append(embedding_unit)
        
        return embedding_units
    
    def compute_summary(self, statements: List[Dict[str, Any]]) -> str:
        """
        Generate a machine-readable synopsis.
        
        Args:
            statements: List of Core Statement dictionaries
        
        Returns:
            Summary string
        """
        if not statements:
            return "No statements extracted."
        
        # Count by type
        type_counts = {}
        for stmt in statements:
            stype = stmt.get('type', 'Unknown')
            type_counts[stype] = type_counts.get(stype, 0) + 1
        
        # Create summary
        summary_parts = [f"Extracted {len(statements)} core statements."]
        summary_parts.append("Type distribution:")
        for stype, count in sorted(type_counts.items()):
            summary_parts.append(f"  - {stype}: {count}")
        
        return " ".join(summary_parts)
    
    def compute_delta(
        self, 
        current_statements: List[Dict[str, Any]],
        prior_index: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute delta compared to prior iteration.
        
        Args:
            current_statements: Current iteration statements
            prior_index: Path to prior iteration index file (optional)
        
        Returns:
            Delta dictionary
        """
        delta = {
            'new': [],
            'changed': [],
            'removed': [],
            'total_current': len(current_statements),
            'total_prior': 0
        }
        
        if not prior_index:
            # First iteration - all statements are new
            delta['new'] = [stmt.get('id') for stmt in current_statements]
            return delta
        
        # Try to load prior index
        try:
            with open(prior_index, 'r', encoding='utf-8') as f:
                prior_data = json.load(f)
            
            prior_statements = prior_data.get('statements', [])
            delta['total_prior'] = len(prior_statements)
            
            # Create maps for comparison
            prior_by_id = {stmt.get('id'): stmt for stmt in prior_statements}
            current_by_id = {stmt.get('id'): stmt for stmt in current_statements}
            
            # Find new statements
            delta['new'] = [
                stmt_id for stmt_id in current_by_id.keys() 
                if stmt_id not in prior_by_id
            ]
            
            # Find removed statements
            delta['removed'] = [
                stmt_id for stmt_id in prior_by_id.keys() 
                if stmt_id not in current_by_id
            ]
            
            # Find changed statements (simplified - just check if text changed)
            for stmt_id in current_by_id:
                if stmt_id in prior_by_id:
                    current_text = current_by_id[stmt_id].get('text', '')
                    prior_text = prior_by_id[stmt_id].get('text', '')
                    if current_text != prior_text:
                        delta['changed'].append(stmt_id)
        
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            # If prior index doesn't exist or is invalid, treat all as new
            delta['new'] = [stmt.get('id') for stmt in current_statements]
            delta['error'] = f"Could not load prior index: {str(e)}"
        
        return delta
    
    def compute_quality_flags(
        self, 
        statements: List[Dict[str, Any]],
        extraction_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute quality flags.
        
        Args:
            statements: List of Core Statement dictionaries
            extraction_results: Results from extraction module
        
        Returns:
            Quality flags dictionary
        """
        flags = {
            'total_statements': len(statements),
            'typed_statements': sum(1 for s in statements if s.get('type')),
            'high_confidence_types': sum(
                1 for s in statements 
                if s.get('type_confidence', 0) >= 0.7
            ),
            'has_risk_targets': len(extraction_results.get('risk_targets', [])) > 0,
            'has_attack_signals': len(extraction_results.get('attack_signal', [])) > 0,
            'has_actions': len(extraction_results.get('action_frames', [])) > 0,
            'warnings': []
        }
        
        # Check for potential issues
        if flags['total_statements'] == 0:
            flags['warnings'].append('No statements extracted')
        
        if flags['typed_statements'] < flags['total_statements']:
            flags['warnings'].append('Some statements lack type classification')
        
        low_confidence_count = flags['total_statements'] - flags['high_confidence_types']
        if low_confidence_count > flags['total_statements'] * 0.5:
            flags['warnings'].append('Many statements have low type confidence')
        
        return flags
    
    def create_provenance(
        self, 
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create provenance metadata.
        
        Args:
            input_data: Original input dictionary
        
        Returns:
            Provenance dictionary
        """
        return {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'prompt_id': input_data.get('prompt_id'),
            'response_id': input_data.get('response_id'),
            'iteration': input_data.get('t'),
            'model': input_data.get('metadata', {}).get('model'),
            'input_timestamp': input_data.get('metadata', {}).get('timestamp')
        }
    
    def structure_output(
        self,
        statements: List[Dict[str, Any]],
        extraction_results: Dict[str, Any],
        input_data: Dict[str, Any],
        prior_index: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Structure the complete output.
        
        Args:
            statements: List of Core Statement dictionaries
            extraction_results: Results from extraction module
            input_data: Original input dictionary
            prior_index: Optional path to prior iteration index
        
        Returns:
            Complete structured output dictionary
        """
        # Create embedding units
        embedding_units = self.create_embedding_units(statements)
        
        # Compute summary
        summary = self.compute_summary(statements)
        
        # Compute delta
        delta = self.compute_delta(statements, prior_index)
        
        # Compute quality flags
        quality_flags = self.compute_quality_flags(statements, extraction_results)
        
        # Create provenance
        provenance = self.create_provenance(input_data)
        
        # Package output
        output = {
            'statements': statements,
            'summary': summary,
            'delta': delta,
            'embedding_units': embedding_units,
            'risk_targets': extraction_results.get('risk_targets', []),
            'action_frames': extraction_results.get('action_frames', []),
            'attack_signal': extraction_results.get('attack_signal', []),
            'quality_flags': quality_flags,
            'provenance': provenance
        }
        
        return output


def structure_output(
    statements: List[Dict[str, Any]],
    extraction_results: Dict[str, Any],
    input_data: Dict[str, Any],
    prior_index: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function for structuring output.
    
    Args:
        statements: List of Core Statement dictionaries
        extraction_results: Results from extraction module
        input_data: Original input dictionary
        prior_index: Optional path to prior iteration index
    
    Returns:
        Complete structured output dictionary
    """
    structurer = Structurer()
    return structurer.structure_output(
        statements, extraction_results, input_data, prior_index
    )



