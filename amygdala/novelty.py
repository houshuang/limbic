"""Backwards-compatibility shim."""
import limbic.amygdala.novelty as _orig
import sys
sys.modules[__name__] = _orig
