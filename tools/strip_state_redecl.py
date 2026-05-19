"""Strip leftover `xxx_state = {...}` re-declarations in data_routes.py.

These shadow the canonical TaskState instances from server/state.py and
break force-stop. Replaces them with an inline comment so existing line
numbers don't shift in diffs.
"""
import os
import re

PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "server", "data_routes.py"))

PATTERN = re.compile(
    r"^([a-z_]+_state)\s*=\s*\{[^}]*\}\s*$",
    re.MULTILINE,
)


def main():
    with open(PATH, "r", encoding="utf-8") as f:
        text = f.read()
    new = PATTERN.sub(
        lambda m: f"# NOTE: {m.group(1)} imported from server.state — do NOT re-declare.",
        text,
    )
    if new != text:
        with open(PATH, "w", encoding="utf-8") as f:
            f.write(new)
        print("UPDATED", PATH)
    else:
        print("unchanged", PATH)


if __name__ == "__main__":
    main()
