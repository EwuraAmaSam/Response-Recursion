"""
Setup script for Iterative Breakdown Layer.
Handles installation and Python 3.12 compatibility fix.
"""

import sys
import subprocess

def main():
    print("=" * 60)
    print("Iterative Breakdown Layer - Setup")
    print("=" * 60)
    print()
    
    # Step 1: Check Python version
    print(f"Python version: {sys.version}")
    if sys.version_info >= (3, 12):
        print("⚠ Python 3.12 detected - compatibility fix may be needed")
    print()
    
    # Step 2: Install dependencies
    print("Step 1: Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed")
    except subprocess.CalledProcessError:
        print("✗ Failed to install dependencies")
        return 1
    print()
    
    # Step 3: Apply Python 3.12 fix if needed
    if sys.version_info >= (3, 12):
        print("Step 2: Applying Python 3.12 compatibility fix...")
        try:
            import fix_pydantic_direct
            print("✓ Compatibility fix applied")
        except Exception as e:
            print(f"⚠ Could not apply fix: {e}")
        print()
    
    # Step 4: Download spaCy model
    print("Step 3: Downloading spaCy model...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✓ spaCy model downloaded")
        else:
            print(f"⚠ Model download had issues:")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
    except Exception as e:
        print(f"⚠ Could not download model: {e}")
    print()
    
    # Step 5: Verify installation
    print("Step 4: Verifying installation...")
    try:
        import spacy
        print(f"✓ spaCy {spacy.__version__} imported successfully")
        
        nlp = spacy.load("en_core_web_sm")
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return 1
    
    print()
    print("=" * 60)
    print("✓ Setup complete! You can now use the IBL.")
    print("=" * 60)
    print()
    print("Try running:")
    print("  python run_ibl.py sample_input_1.json output.json")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())



