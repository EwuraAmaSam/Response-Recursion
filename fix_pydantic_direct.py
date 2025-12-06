"""
Direct fix for pydantic v1 Python 3.12 compatibility.
This patches the actual pydantic v1 file.
"""

import sys
import os

if sys.version_info >= (3, 12):
    try:
        import pydantic.v1.typing as pydantic_v1_typing
        
        # Get the file path
        pydantic_v1_typing_file = pydantic_v1_typing.__file__
        
        # Read the file
        with open(pydantic_v1_typing_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if already patched
        if 'recursive_guard=set()' in content or 'recursive_guard: set' in content:
            print("✓ pydantic v1 already patched or compatible")
        else:
            # Patch the evaluate_forwardref function
            old_line = "    return cast(Any, type_)._evaluate(globalns, localns, set())"
            new_line = "    return cast(Any, type_)._evaluate(globalns, localns, recursive_guard=set())"
            
            if old_line in content:
                content = content.replace(old_line, new_line)
                
                # Write back
                with open(pydantic_v1_typing_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"✓ Patched {pydantic_v1_typing_file}")
                print("  You may need to restart Python for changes to take effect.")
            else:
                print("⚠ Could not find the line to patch. The file may have a different structure.")
                print(f"  File: {pydantic_v1_typing_file}")
    except Exception as e:
        print(f"Error applying patch: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Applying pydantic v1 Python 3.12 compatibility patch...")
    # The patch code above runs on import



