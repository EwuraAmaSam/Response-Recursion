#!/usr/bin/env python
"""
Simple runner script for the Iterative Breakdown Layer.
Can be run from the project root directory.
"""

import sys
import json
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from iterative_breakdown_layer import IterativeBreakdownLayer, process_input


def main():
    """Command-line interface."""
    if len(sys.argv) < 2:
        print("Usage: python run_ibl.py <input_file> [output_file]")
        print("\nExample:")
        print("  python run_ibl.py sample_input_1.json output.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        output = process_input(input_file, output_file)
        
        if output_file:
            print(f"✓ Successfully processed {input_file}")
            print(f"✓ Output saved to {output_file}")
            print(f"\nSummary: {output.get('summary', 'N/A')}")
            print(f"Statements extracted: {len(output.get('statements', []))}")
        else:
            # Print JSON output
            print(json.dumps(output, indent=2, ensure_ascii=False))
    
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()



