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
        """Register the mouse mode and hook ViewDock if it is running."""
        try:
            from .mousemode import register_mouse_mode
            register_mouse_mode(session)
        except Exception as e:
            session.logger.warning("hitpick: mouse mode unavailable: %s" % e)
        try:
            from .viewdock_link import autohook
            autohook(session)
        except Exception as e:
            session.logger.info("hitpick: ViewDock link not established (%s)" % e)

    @staticmethod
    def finish(session, bundle_info):
        try:
            from .viewdock_link import unhook
            unhook(session)
        except Exception:
            pass
        from .state import clear_state
        clear_state(session)

    @staticmethod
    def get_class(class_name):
        if class_name == "HandbookTool":
            from .tool import HandbookTool
            return HandbookTool
        return None


bundle_api = _HitPickAPI()
