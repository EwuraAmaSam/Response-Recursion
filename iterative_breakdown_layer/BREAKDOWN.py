"""
Main entry point for the Iterative Breakdown Layer (IBL).

This module orchestrates the complete pipeline:
1. Preprocessing
2. Sentence Segmentation
3. Clause Decomposition
4. Semantic Merge/Split
5. Statement Typing
6. Targeted Extraction
7. Structuring
"""

import json
import sys
from typing import Dict, Any, Optional
from pathlib import Path

# Import pipeline modules
from .pipeline.preprocessing import preprocess_text, restore_code_fences
from .pipeline.segmentation import SentenceSegmenter
from .pipeline.clause_decomposition import ClauseDecomposer
from .pipeline.semantic_merge import SemanticMerger
from .pipeline.typing import StatementTyper
from .pipeline.extraction import Extractor
from .pipeline.structuring import Structurer
from .utils.nlp import get_nlp_model


class IterativeBreakdownLayer:
    """Main IBL class that orchestrates the pipeline."""
    
    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        embedding_model: str = "all-mpnet-base-v2",
        use_ml_typing: bool = False,
        ml_typing_model: Optional[str] = None
    ):
        """
        Initialize the IBL.
        
        Args:
            spacy_model: spaCy model name
            embedding_model: Sentence transformer model name
            use_ml_typing: Whether to use ML for statement typing
            ml_typing_model: Optional ML model for typing
        """
        # Load spaCy model (shared across modules)
        self.nlp = get_nlp_model(spacy_model)
        
        # Initialize pipeline components
        self.segmenter = SentenceSegmenter(model_name=spacy_model)
        self.decomposer = ClauseDecomposer(nlp_model=self.nlp)
        self.merger = SemanticMerger(model_name=embedding_model)
        self.typer = StatementTyper(use_ml=use_ml_typing, model_name=ml_typing_model)
        self.extractor = Extractor(nlp_model=self.nlp)
        self.structurer = Structurer()
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input through the complete pipeline.
        
        Args:
            input_data: Input dictionary with:
                - t: iteration number
                - prompt_id: prompt identifier
                - response_id: response identifier
                - prior_index: path to prior iteration index (optional)
                - prompt_text: prompt text
                - response_text: response text
                - metadata: additional metadata
        
        Returns:
            Complete structured output dictionary
        """
        # Step 0: Preprocessing
        preprocessed = preprocess_text(
            input_data.get('response_text', ''),
            preserve_structure=True
        )
        preprocessed_text = preprocessed['preprocessed_text']
        placeholders = preprocessed['placeholders']
        
        # Step 1: Sentence Segmentation
        sentences = self.segmenter.segment(preprocessed_text)
        
        if not sentences:
            # Empty response - return minimal output
            return self._create_empty_output(input_data)
        
        # Step 2: Clause Decomposition
        clauses = self.decomposer.decompose_sentences(sentences)
        
        # Step 3: Semantic Merge/Split
        core_statements = self.merger.process_clauses(clauses)
        
        # Step 4: Statement Typing
        typed_statements = self.typer.type_statements(core_statements)
        
        # Step 5: Targeted Extraction
        extraction_results = self.extractor.extract_all(typed_statements)
        
        # Step 6: Structuring
        prior_index = input_data.get('prior_index')
        output = self.structurer.structure_output(
            typed_statements,
            extraction_results,
            input_data,
            prior_index
        )
        
        # Restore code fences in statement texts if needed
        if placeholders:
            for statement in output['statements']:
                statement['text'] = restore_code_fences(
                    statement['text'],
                    placeholders
                )
            for unit in output['embedding_units']:
                unit['clean_text'] = restore_code_fences(
                    unit['clean_text'],
                    placeholders
                )
        
        return output
    
    def _create_empty_output(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create output for empty response."""
        from datetime import datetime
        
        provenance = self.structurer.create_provenance(input_data)
        
        return {
            'statements': [],
            'summary': 'No statements extracted from empty response.',
            'delta': {
                'new': [],
                'changed': [],
                'removed': [],
                'total_current': 0,
                'total_prior': 0
            },
            'embedding_units': [],
            'risk_targets': [],
            'action_frames': [],
            'attack_signal': [],
            'quality_flags': {
                'total_statements': 0,
                'typed_statements': 0,
                'high_confidence_types': 0,
                'has_risk_targets': False,
                'has_attack_signals': False,
                'has_actions': False,
                'warnings': ['Empty response text']
            },
            'provenance': provenance
        }


def process_input(input_file: str, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Process input from a JSON file.
    
    Args:
        input_file: Path to input JSON file
        output_file: Optional path to output JSON file
    
    Returns:
        Processed output dictionary
    """
    # Load input
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    # Process
    ibl = IterativeBreakdownLayer()
    output = ibl.process(input_data)
    
    # Save output if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    
    return output


def main():
    """Command-line interface."""
    if len(sys.argv) < 2:
        print("Usage: python BREAKDOWN.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        output = process_input(input_file, output_file)
        print(json.dumps(output, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

