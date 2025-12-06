# Iterative Breakdown Layer (IBL)

The Iterative Breakdown Layer (IBL) is a semantic processing module responsible for converting raw LLM responses into structured, testable meaning units. It acts as a deterministic "semantic compiler" whose outputs feed directly into embedding systems and prompt generation systems.

## Features

- **Atomic Semantic Units**: Breaks responses into Core Statements (CS)
- **Statement Classification**: Classifies statements into types (Instruction, Conditional, Warning, etc.)
- **Pattern Extraction**: Extracts entities, actions, logical dependencies, uncertainty cues, and attack signals
- **Iteration Comparison**: Compares iterations for drift, novelty, and contradictions
- **Structured Output**: Produces embedding profiles and prompt-generation profiles

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

For better performance, use the transformer-based model:
```bash
python -m spacy download en_core_web_trf
```

## Usage

### Python API

```python
from iterative_breakdown_layer import IterativeBreakdownLayer

# Initialize IBL
ibl = IterativeBreakdownLayer()

# Prepare input
input_data = {
    "t": 1,
    "prompt_id": "p_01",
    "response_id": "r_01",
    "prior_index": None,
    "prompt_text": "Explain how nations can prepare for biological threats responsibly.",
    "response_text": "Nations can improve preparedness by investing in early detection systems...",
    "metadata": {
        "model": "gpt-5",
        "timestamp": "2025-12-01T20:07:00Z"
    }
}

# Process
output = ibl.process(input_data)

# Access results
print(output['statements'])
print(output['risk_targets'])
print(output['attack_signal'])
```

### Command Line

```bash
python iterative_breakdown_layer/BREAKDOWN.py input.json output.json
```

## Input Format

```json
{
  "t": 1,
  "prompt_id": "p_01",
  "response_id": "r_01",
  "prior_index": "iteration_1_index.json",
  "prompt_text": "Your prompt here",
  "response_text": "LLM response text here",
  "metadata": {
    "model": "gpt-5",
    "timestamp": "2025-12-01T20:07:00Z"
  }
}
```

## Output Format

The IBL produces a structured JSON output with:

- **statements**: List of Core Statements with types, spans, and metadata
- **summary**: Machine-readable synopsis
- **delta**: Comparison with prior iteration (new/changed/removed statements)
- **embedding_units**: Cleaned text units for embedding
- **risk_targets**: Extracted risk-relevant entities and domains
- **action_frames**: Extracted action patterns
- **attack_signal**: Detected jailbreak/attack patterns
- **quality_flags**: Quality metrics and warnings
- **provenance**: Metadata about the processing

## Pipeline Architecture

1. **Preprocessing**: Normalizes whitespace, preserves code/quotations
2. **Sentence Segmentation**: Uses spaCy for sentence boundary detection
3. **Clause Decomposition**: Breaks sentences into evaluable clauses using dependency parsing
4. **Semantic Merge/Split**: Recombines fragments and splits multi-proposition clauses
5. **Statement Typing**: Hybrid rule-based + ML classification
6. **Targeted Extraction**: Extracts entities, actions, logic, uncertainty, attack signals
7. **Structuring**: Packages outputs into embedding and prompt-generation profiles

## Project Structure

```
iterative_breakdown_layer/
├── pipeline/
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── clause_decomposition.py
│   ├── semantic_merge.py
│   ├── typing.py
│   ├── extraction.py
│   └── structuring.py
├── utils/
│   ├── spans.py
│   ├── nlp.py
│   ├── patterns.py
│   └── delta.py
├── BREAKDOWN.py
└── __init__.py
```

## Dependencies

- spaCy >= 3.7.0
- transformers >= 4.30.0
- sentence-transformers >= 2.2.0
- scikit-learn >= 1.3.0
- numpy >= 1.24.0
- scipy >= 1.10.0

## Notes

- The IBL does not include the embedding system or candidate prompt generator (these are separate components)
- The module is designed to be deterministic and reproducible
- All outputs preserve links back to original text spans
- Supports iteration comparison via prior_index parameter

## License

See LICENSE file for details.



