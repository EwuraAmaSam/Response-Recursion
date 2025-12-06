# Fixing spaCy Python 3.12 Compatibility Issue

## Problem

You're encountering this error:
```
TypeError: ForwardRef._evaluate() missing 1 required keyword-only argument: 'recursive_guard'
```

This is a compatibility issue between spaCy 3.7.x, Pydantic v1, and Python 3.12. Python 3.12 changed some internal APIs that Pydantic v1 relies on.

## Solution

### Option 1: Upgrade to spaCy 3.8+ (Best Solution)

spaCy 3.8+ has better Python 3.12 support. Try this:

```bash
pip install --upgrade "spacy>=3.8.0"
python -m spacy download en_core_web_sm
```

If spaCy 3.8+ is not available yet, use Option 2.

### Option 2: Fix Pydantic Version (If Option 1 doesn't work)

Pin Pydantic to a compatible version:

```bash
pip install "pydantic>=1.10.13,<2.0.0"
pip install --upgrade --force-reinstall spacy
python -m spacy download en_core_web_sm
```

### Option 3: Use Python 3.11 (Most Reliable)

If the above don't work, Python 3.11 is fully compatible:

```bash
# Create a virtual environment with Python 3.11
# First, install Python 3.11 if you don't have it
python3.11 -m venv venv
venv\Scripts\activate  # On Windows PowerShell
# or
venv\Scripts\activate.bat  # On Windows CMD

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Option 4: Quick Fix - Reinstall with Specific Versions

Try this sequence:

```bash
pip uninstall spacy pydantic -y
pip install "pydantic>=1.10.13,<2.0.0"
pip install "spacy>=3.7.0,<3.8.0"
python -m spacy download en_core_web_sm
```

## Verification

After fixing, verify it works:

```bash
python -c "import spacy; print('spaCy version:', spacy.__version__)"
python -m spacy download en_core_web_sm
```

If the download command completes without errors, you're good to go!

## Note

The requirements.txt has been updated to pin compatible versions. After updating requirements.txt, run:

```bash
pip install --upgrade -r requirements.txt
python -m spacy download en_core_web_sm
```

