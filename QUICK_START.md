# Quick Start Guide - How to Run the IBL

## ✅ Prerequisites (One-Time Setup)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Fix Python 3.12 compatibility (if using Python 3.12):**
   ```bash
   python fix_pydantic_direct.py
   ```

3. **Install tf-keras (for transformers compatibility):**
   ```bash
   pip install tf-keras
   ```

4. **Download spaCy model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

## 🚀 Running the Code

### Method 1: Using the Runner Script (Easiest)

```bash
python run_ibl.py sample_input_1.json output.json
```

This will:
- Process `sample_input_1.json`
- Save results to `output.json`
- Print a summary

### Method 2: Test All Sample Inputs

```bash
python test_samples.py
```

This processes all 3 sample inputs and shows detailed results.

### Method 3: Python API

```python
from iterative_breakdown_layer import IterativeBreakdownLayer
import json

# Initialize
ibl = IterativeBreakdownLayer()

# Your input
input_data = {
    "t": 1,
    "prompt_id": "p_01",
    "response_id": "r_01",
    "prior_index": None,
    "prompt_text": "Your prompt here",
    "response_text": "LLM response text here",
    "metadata": {
        "model": "gpt-5",
        "timestamp": "2025-12-01T20:07:00Z"
    }
}

# Process
output = ibl.process(input_data)

# Save results
with open('output.json', 'w') as f:
    json.dump(output, f, indent=2)
```

## 📋 Input Format

Your input JSON should have this structure:

```json
{
  "t": 1,
  "prompt_id": "p_01",
  "response_id": "r_01",
  "prior_index": null,
  "prompt_text": "Your prompt text",
  "response_text": "LLM response text",
  "metadata": {
    "model": "gpt-5",
    "timestamp": "2025-12-01T20:07:00Z"
  }
}
```

## 📤 Output

The output JSON contains:
- **statements**: Extracted core statements with types
- **risk_targets**: Risk-relevant entities
- **attack_signal**: Detected jailbreak patterns
- **embedding_units**: Cleaned text for embeddings
- **delta**: Comparison with prior iteration
- **quality_flags**: Quality metrics

## ⚠️ Troubleshooting

### First Run Takes Time
The first time you run it, it will download the sentence transformer model (~438MB). This is normal and only happens once.

### TensorFlow Warnings
You may see TensorFlow warnings about oneDNN. These are harmless and can be ignored.

### If You Get Import Errors
Make sure you've:
1. Installed all dependencies: `pip install -r requirements.txt`
2. Installed tf-keras: `pip install tf-keras`
3. Applied the Python 3.12 fix: `python fix_pydantic_direct.py`

## 🎯 Example

```bash
# Run with sample input
python run_ibl.py sample_input_1.json output.json

# Check the output
# Open output.json to see the structured results
```

That's it! The IBL is now ready to use. 🎉


