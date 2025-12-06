"""
Monkey patch to fix pydantic v1 compatibility with Python 3.12.
This must be imported BEFORE importing spacy.
"""

import sys
import typing

# Check if we're on Python 3.12+
if sys.version_info >= (3, 12):
    try:
        from pydantic.v1 import typing as pydantic_v1_typing
        
        # Patch the evaluate_forwardref function to work with Python 3.12
        original_evaluate_forwardref = pydantic_v1_typing.evaluate_forwardref
        
        def patched_evaluate_forwardref(field_type, globalns=None, localns=None):
            """Patched version that works with Python 3.12."""
            try:
                # Try the new Python 3.12 API first
                if hasattr(field_type, '_evaluate'):
                    try:
                        return field_type._evaluate(globalns, localns, recursive_guard=set())
                    except TypeError:
                        # Fall back to old API if new one doesn't work
                        pass
                # Fall back to original function
                return original_evaluate_forwardref(field_type, globalns, localns)
            except Exception:
                # If all else fails, return the type as-is
                return field_type
        
        # Apply the patch
        pydantic_v1_typing.evaluate_forwardref = patched_evaluate_forwardref
        
        print("✓ Applied Python 3.12 compatibility patch for pydantic v1")
    except ImportError:
        # pydantic v1 not found, no patch needed
        pass
    except Exception as e:
        print(f"Warning: Could not apply pydantic patch: {e}")



