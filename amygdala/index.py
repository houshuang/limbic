"""Backwards-compatibility shim."""
import limbic.amygdala.index as _orig
import sys
sys.modules[__name__] = _orig
