# Quick Start Guide

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Basic Usage

### Python API

```python
from iterative_breakdown_layer import IterativeBreakdownLayer

# Initialize
ibl = IterativeBreakdownLayer()

# Prepare input
input_data = {
    "t": 1,
    "prompt_id": "p_01",
    "response_id": "r_01",
    "prior_index": None,
    "prompt_text": "Your prompt",
    "response_text": "LLM response text",
    "metadata": {
        "model": "gpt-5",
        "timestamp": "2025-12-01T20:07:00Z"
    }
}

# Process
output = ibl.process(input_data)

# Access results
print(f"Found {len(output['statements'])} statements")
print(f"Risk targets: {len(output['risk_targets'])}")
print(f"Attack signals: {len(output['attack_signal'])}")
```

### Command Line

```bash
python iterative_breakdown_layer/BREAKDOWN.py sample_input_1.json output.json
```

### Test Script

Run the test script with sample inputs:

```bash
python test_samples.py
```

## Output Structure

The IBL produces a JSON output with:

- **statements**: Core Statements with types, spans, confidence scores
- **embedding_units**: Cleaned text for embedding systems
- **risk_targets**: Extracted risk-relevant entities
- **action_frames**: Action patterns (Planning, Bypassing, etc.)
- **attack_signal**: Detected jailbreak/attack patterns
- **delta**: Comparison with prior iteration
- **quality_flags**: Quality metrics

## Statement Types

- Instruction
- Conditional
- Warning/Policy
- Definition
- Assertion/Fact
- Meta-safety
- Refusal
- Hedge/Uncertainty

## Notes

- The IBL is deterministic and reproducible
- All outputs preserve character spans back to original text
- Supports iteration comparison via `prior_index` parameter
- Embedding system and prompt generator are separate components



