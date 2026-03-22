"""Backwards-compatibility shim."""
import limbic.amygdala.llm as _orig
import sys
sys.modules[__name__] = _orig
