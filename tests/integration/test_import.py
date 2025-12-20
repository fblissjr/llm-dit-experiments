#!/usr/bin/env python
"""Quick import test for diffusers from coderef."""

import sys
from pathlib import Path

print("Python:", sys.version)
print("Path before:", sys.path[:3])

# Add coderef diffusers to path
coderef_diffusers = Path(__file__).parent.parent.parent / "coderef" / "diffusers" / "src"
print(f"Adding coderef: {coderef_diffusers}")
print(f"Exists: {coderef_diffusers.exists()}")
sys.path.insert(0, str(coderef_diffusers))

try:
    print("Importing diffusers...")
    import diffusers
    print(f"diffusers version: {diffusers.__version__}")
    print(f"diffusers location: {diffusers.__file__}")

    print("Checking for QwenImageLayeredPipeline...")
    from diffusers import QwenImageLayeredPipeline
    print("SUCCESS: QwenImageLayeredPipeline imported!")

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
