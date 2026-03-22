"""Backwards-compatibility shim."""
import limbic.amygdala.document_similarity as _orig
import sys
sys.modules[__name__] = _orig
