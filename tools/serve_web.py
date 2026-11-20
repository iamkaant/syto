#!/usr/bin/env python
"""Serve web/ and open it. One command, because there has to be one.

The page cannot run from file://: module workers and WebAssembly are both
blocked there, which is a browser rule and not something the page can work
around. So it needs a server, and asking a chemist to remember an
incantation to look at a pose is how a tool stops being used.

Nothing is uploaded anywhere. This serves the folder to this machine only,
on the loopback address, and the receptor and poses are read by the page
in the tab - they never cross the network at all.

Usage:  python tools/serve_web.py [--port 8000] [--no-browser]
"""
from __future__ import annotations

import argparse
import functools
import http.server
import os
import sys
import webbrowser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    missing = [p for p in ("vendor/pyodide/pyodide.mjs",
                           "vendor/molstar/molstar.js",
                           "vendor/hitpicking.zip")
               if not os.path.exists(os.path.join(WEB, p))]
    if missing:
        print("web/ is not built yet - missing:")
        for m in missing:
            print("   ", m)
        print("\nRun:\n  python tools/fetch_web_deps.py"
              "\n  python tools/build_web_core.py")
        return 1

    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=WEB)
    # 127.0.0.1, not 0.0.0.0: this is for the person at this keyboard, and
    # binding every interface would put a file browser of this directory on
    # whatever network the machine happens to be on.
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = "http://127.0.0.1:%d/" % args.port
    print("Hit Picking is at %s   (ctrl-C to stop)" % url)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
