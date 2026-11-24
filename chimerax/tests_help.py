"""Help is wired up: the tools' Help menu entry and `help hitpick ...`.

The ChimeraX Toolshed reviewers asked for it. What is pinned here is that
each tool's `help` names a page ChimeraX can actually find, that every
registered hitpick command resolves to the command page, and that the
anchor it points at exists - a help link that opens the top of the page
for one command and the right paragraph for another is how these rot.

Run:  ChimeraX --nogui --silent --exit --script chimerax/tests_help.py
"""
import os
import re

from chimerax.core import toolshed
from chimerax.core.commands import cli, run

failures = []


def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def find(help_url):
    """The file a help: URL resolves to, the way the help command looks."""
    rel = help_url[len("help:"):].split("#")[0]
    for hd in toolshed.get_help_directories():
        p = os.path.join(hd, rel)
        if os.path.exists(p):
            return p
    return None


def anchors(path):
    with open(path, encoding="utf-8") as fh:
        return set(re.findall(r'<a name="([^"]+)"', fh.read()))


run(session, "hitpick clear", log=False)      # registers every command

from chimerax.syto.tool import HandbookTool
from chimerax.syto.learn_tool import LearnTool

for cls in (HandbookTool, LearnTool):
    h = cls.help
    check(bool(h) and h.startswith("help:"), "%s.help is set (%s)"
          % (cls.__name__, h))
    p = find(h) if h else None
    check(p is not None, "%s help page is installed" % cls.__name__)
    if p:
        text = open(p, encoding="utf-8").read()
        for link in re.findall(r'href="\.\./(tools|commands)/(syto_\w+|hitpick)'
                               r'\.html(?:#([\w-]+))?"', text):
            target = find("help:user/%s/%s.html" % link[:2])
            check(target is not None and
                  (not link[2] or link[2] in anchors(target)),
                  "%s links to %s/%s#%s, which exists"
                  % (cls.__name__, link[0], link[1], link[2]))

names = sorted(n for n in cli.registered_commands(multiword=True)
               if n.startswith("hitpick "))
check(len(names) >= 35, "%d hitpick commands are registered" % len(names))
page = find("help:user/commands/hitpick.html")
check(page is not None, "the command page is installed")
have = anchors(page) if page else set()
for n in names:
    try:
        url = cli.command_url(n)
    except ValueError:
        url = None
    frag = url.split("#")[1] if url and "#" in url else None
    check(bool(url) and url.startswith("help:user/commands/hitpick.html")
          and frag in have, "help %s -> %s" % (n, url))

# Every command the bundle registers must also be declared, or ChimeraX
# does not know it until some other hitpick command has been typed.
bi = [b for b in session.toolshed.bundle_info(session.logger)
      if b.name == "ChimeraX-Syto"]
check(len(bi) == 1, "the bundle is installed")
if bi:
    declared = set(c.name for c in bi[0].commands)
    missing = [n for n in names if n not in declared]
    check(not missing, "every registered command is declared in "
          "pyproject.toml (missing: %s)" % missing)

if failures:
    print("FAILURES (%d):" % len(failures))
    for f in failures:
        print("   -", f)
    raise SystemExit(1)
print("HELP TESTS OK")
