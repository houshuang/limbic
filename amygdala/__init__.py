"""Backwards-compatibility shim: `from amygdala import X` still works.

Delegates to limbic.amygdala. Will be removed once all consumers migrate
to `from limbic.amygdala import X`.
"""

from limbic.amygdala import *  # noqa: F401,F403
from limbic.amygdala import __all__  # noqa: F401
