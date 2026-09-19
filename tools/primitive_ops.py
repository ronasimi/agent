"""Compatibility facade for Unix-style primitives.

New code should import from ``tools.primitive_modules.<domain>``.  This module
keeps the historical import path stable for recipes, tests, and external tools.
"""
from .primitive_modules.filesystem import *  # noqa: F401,F403
from .primitive_modules.text import *  # noqa: F401,F403
from .primitive_modules.structured import *  # noqa: F401,F403
from .primitive_modules.process import *  # noqa: F401,F403
from .primitive_modules.system import *  # noqa: F401,F403
from .primitive_modules.network import *  # noqa: F401,F403
from .primitive_modules.web import *  # noqa: F401,F403
from .primitive_modules.documents import *  # noqa: F401,F403
from .primitive_modules.media import *  # noqa: F401,F403
from .primitive_modules.archive import *  # noqa: F401,F403
from .primitive_modules.git import *  # noqa: F401,F403
from .primitive_modules.database import *  # noqa: F401,F403
from .primitive_modules.utility import *  # noqa: F401,F403
from .primitive_modules.observation import *  # noqa: F401,F403
