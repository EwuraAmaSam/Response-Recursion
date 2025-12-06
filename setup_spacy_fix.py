"""
Setup script to fix spaCy Python 3.12 compatibility issue.
Run this before using spaCy.
"""

import fix_pydantic_python312  # Apply the patch
import spacy

print(f"spaCy version: {spacy.__version__}")
print("✓ spaCy imported successfully!")

# Try to download the model
try:
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("✓ Model downloaded successfully!")
    else:
        print(f"Model download output: {result.stdout}")
        if result.stderr:
            print(f"Errors: {result.stderr}")
except Exception as e:
    print(f"Error downloading model: {e}")



