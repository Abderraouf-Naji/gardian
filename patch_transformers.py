# patch_transformers.py
"""Patch transformers to disable auto-conversion."""

import os
os.environ["HF_DISABLE_SAFETENSORS_CONVERSION"] = "1"
os.environ["SAFETENSORS_FAST_GPU"] = "0"

import sys
import transformers
from transformers import modeling_utils

def disabled_auto_conversion(*args, **kwargs):
    """Completely disable auto-conversion."""
    return None

# Apply the patch
if hasattr(modeling_utils, 'auto_conversion'):
    modeling_utils.auto_conversion = disabled_auto_conversion
    print("✓ Patched transformers.auto_conversion")

# Also patch at module level
if hasattr(transformers.modeling_utils, 'auto_conversion'):
    transformers.modeling_utils.auto_conversion = disabled_auto_conversion
    print("✓ Patched transformers.modeling_utils.auto_conversion")

print("✓ Transformers patched successfully")