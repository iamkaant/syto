"""HitPickingTool: a hit-picking helper for ChimeraX.

Rides alongside ViewDock. The chemistry, the interaction scoring and the
handbook all come from the `hitpicking` core package, which is shared
verbatim with the web front end; nothing in this bundle duplicates it.
"""

import os
import sys

# The shared engine ships vendored inside this bundle (tools/vendor_core.py),
# so installing the bundle is the only step a user needs. It is put on the
# path before anything imports it, and only if the user has not already
# installed their own copy - a developer working on the core will have it
# pip-installed and should get theirs, not ours.
_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor")
if os.path.isdir(_VENDOR):
    try:
        import hitpicking  # noqa: F401
    except ImportError:
        sys.path.append(_VENDOR)

from chimerax.core.toolshed import BundleAPI

__version__ = "0.1.0"


class _HitPickAPI(BundleAPI):

    api_version = 1

    @staticmethod
    def start_tool(session, bundle_info, tool_info, **kw):
        if tool_info.name == "Hit Picking Handbook":
            from .tool import HandbookTool
            return HandbookTool(session, tool_info.name)
        raise ValueError("unknown tool %r" % tool_info.name)

    @staticmethod
    def register_command(bundle_info, command_info, logger):
        from . import cmd
        cmd.register_commands(logger)

    @staticmethod
    def initialize(session, bundle_info):
        """Deliberately does nothing.

        custom-init is false in pyproject.toml, so this should never be
        called. It exists because ChimeraX caches bundle metadata: an older
        cached entry can still say custom-init=true, and a missing initialize()
        then reports "custom initialization failed" on every startup. Doing
        nothing here is safe under both.

        Nothing may be added to this method. Touching session.ui at startup is
        what caused the duplicate 'mouse hover' trigger; setup is lazy, in
        cmd.ensure_ui.
        """
        return

    # NOTE: the real setup deliberately does NOT happen at startup. Registering the mouse mode
    # or hooking ViewDock at startup means touching session.ui before ChimeraX
    # has finished building it, which constructs the mouse-mode manager early
    # and leaves ObjectIdMouseMode trying to add the 'mouse hover' trigger a
    # second time - breaking the Toolbar tool. Both are set up lazily instead,
    # on the first `hitpick` command or when the panel opens (see cmd.ensure_ui).

    @staticmethod
    def get_class(class_name):
        if class_name == "HandbookTool":
            from .tool import HandbookTool
            return HandbookTool
        return None


bundle_api = _HitPickAPI()
