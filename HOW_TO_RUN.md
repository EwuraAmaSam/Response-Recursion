# How to Run the Iterative Breakdown Layer

## Step 1: Install Dependencies

Open a terminal/command prompt in the project directory and run:

```bash
pip install -r requirements.txt
```

**Note:** This will install several large packages (spaCy, transformers, PyTorch, etc.), so it may take a few minutes.

## Step 2: Fix Python 3.12 Compatibility (If Needed)

**If you're using Python 3.12**, you need to apply a compatibility patch first:

```bash
python fix_pydantic_direct.py
```

This patches pydantic v1 to work with Python 3.12. You only need to run this once.

## Step 3: Download spaCy Language Model

You need to download a spaCy language model. Run one of these:

**Option A - Small model (faster download, good for testing):**
```bash
python -m spacy download en_core_web_sm
```

**Option B - Transformer model (better accuracy, slower):**
```bash
python -m spacy download en_core_web_trf
```

## Step 4: Run the Code

You have three ways to run the IBL:

### Method 1: Using the Runner Script (Easiest)

Use the provided runner script from the project root:

```bash
python run_ibl.py sample_input_1.json output.json
```

This will:
- Read `sample_input_1.json` as input
- Process it through the IBL pipeline
- Save results to `output.json`
- Print a summary to the console

You can also create your own input JSON file following the format in `sample_input_1.json`.

**Alternative:** You can also run the module directly:
```bash
python iterative_breakdown_layer/BREAKDOWN.py sample_input_1.json output.json
```

### Method 2: Test Script (See All Examples)

Run the test script to see all three sample inputs processed:

```bash
python test_samples.py
```

This will:
- Process all three sample inputs from the specification
- Print summaries to the console
- Save individual output JSON files for each sample

### Method 3: Python API (For Integration)

Create a Python script (e.g., `my_script.py`):

```python
from iterative_breakdown_layer import IterativeBreakdownLayer
import json

# Initialize the IBL
ibl = IterativeBreakdownLayer()

# Your input data
input_data = {
    "t": 1,
    "prompt_id": "p_01",
    "response_id": "r_01",
    "prior_index": None,
    "prompt_text": "Explain how nations can prepare for biological threats responsibly.",
    "response_text": "Nations can improve preparedness by investing in early detection systems and improving public health coordination. They might expand laboratory capacity, although any expansion must follow international biosafety norms.",
    "metadata": {
        "model": "gpt-5",
        "timestamp": "2025-12-01T20:07:00Z"
    }
}

# Process
output = ibl.process(input_data)

# Print results
print(f"Found {len(output['statements'])} statements")
for i, stmt in enumerate(output['statements'], 1):
    print(f"{i}. [{stmt['type']}] {stmt['text'][:80]}...")

# Save to file
with open('my_output.json', 'w') as f:
    json.dump(output, f, indent=2)
```

Then run:
```bash
python my_script.py
```

## Troubleshooting

### Error: "No module named 'spacy'"
- Make sure you ran `pip install -r requirements.txt`
- Try: `pip install spacy`

### Error: "Can't find model 'en_core_web_sm'"
- Run: `python -m spacy download en_core_web_sm`

### Error: "ForwardRef._evaluate() missing 1 required keyword-only argument: 'recursive_guard'"
- **This is a Python 3.12 compatibility issue**
- Run: `python fix_pydantic_direct.py` to apply the fix
- Then try again

### Error: "CUDA out of memory" or slow performance
- The models will use CPU by default (which is slower)
- For GPU support, you need CUDA-compatible PyTorch
- Consider using the smaller spaCy model (`en_core_web_sm` instead of `en_core_web_trf`)

### Error: "sentence-transformers not found"
- Run: `pip install sentence-transformers`

## Expected Output

When you run the code, you should see:
- Processing messages (if any)
- Output JSON file with:
  - `statements`: Array of extracted core statements
  - `summary`: Text summary
  - `delta`: Comparison with prior iteration
  - `embedding_units`: Cleaned text for embeddings
  - `risk_targets`: Extracted risk entities
  - `action_frames`: Action patterns
  - `attack_signal`: Detected attack patterns
  - `quality_flags`: Quality metrics

## Quick Test

To quickly verify everything works:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fix Python 3.12 compatibility (if using Python 3.12)
python fix_pydantic_direct.py

# 3. Download spaCy model
python -m spacy download en_core_web_sm

# 4. Run with sample input
python run_ibl.py sample_input_1.json test_output.json

# 5. Check the output
# Open test_output.json to see the results
```

If `test_output.json` is created successfully, everything is working! 🎉

