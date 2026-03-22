"""Backwards-compatibility shim."""
import limbic.amygdala.cache as _orig
import sys
sys.modules[__name__] = _orig
