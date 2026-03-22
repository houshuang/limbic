"""Backwards-compatibility shim."""
import limbic.amygdala.knowledge_map_gen as _orig
import sys
sys.modules[__name__] = _orig
