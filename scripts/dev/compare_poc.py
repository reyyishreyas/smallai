"""Compare two Stage A payloads, ignoring wall-clock fields.

Development tool: used to prove that a change to the training code
left the scored Stage A run untouched. The comparison walks both
JSON documents and reports every leaf that differs, skipping the
fields that are wall-clock by nature (latency, elapsed time) and
therefore cannot be identical between two runs.

Usage::

    python scripts/dev/compare_poc.py reports/poc.json other/poc.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

#: Leaf names whose values are wall-clock measurements.
VOLATILE = re.compile(
    r"lat|elaps|second|_ms$|duration|clock|throughput|wall",
    re.IGNORECASE,
)


def flatten(node, prefix: str = "") -> dict[str, object]:
    """Flatten a JSON document into ``path -> leaf value``.

    Parameters
    ----------
    node : object
        The JSON value.
    prefix : str
        Path prefix for the current node.

    Returns
    -------
    dict
        Every leaf, keyed by its dotted path.
    """
    out: dict[str, object] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            out.update(flatten(value, f"{prefix}.{key}" if prefix else key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            out.update(flatten(value, f"{prefix}[{index}]"))
    else:
        out[prefix] = node
    return out


def main(argv: list[str]) -> int:
    """Compare two payloads and report the differences."""
    if len(argv) != 3:
        print(__doc__)
        return 2
    left = flatten(json.loads(Path(argv[1]).read_text(encoding="utf-8")))
    right = flatten(json.loads(Path(argv[2]).read_text(encoding="utf-8")))

    only_left = sorted(set(left) - set(right))
    only_right = sorted(set(right) - set(left))
    differing = sorted(
        key
        for key in set(left) & set(right)
        if left[key] != right[key]
    )
    volatile = [key for key in differing if VOLATILE.search(key)]

    print(f"leaves: {len(left)} vs {len(right)}")
    print(f"only in {argv[1]}: {len(only_left)}")
    for key in only_left[:20]:
        print(f"  - {key}")
    print(f"only in {argv[2]}: {len(only_right)}")
    for key in only_right[:20]:
        print(f"  + {key}")
    print(f"differing: {len(differing)} (of which wall-clock: {len(volatile)})")
    for key in differing:
        if VOLATILE.search(key):
            print(f"  ~ {key}: {left[key]!r} vs {right[key]!r}  (wall-clock)")
        else:
            print(f"  ! {key}: {left[key]!r} vs {right[key]!r}")
    hard = [key for key in differing if not VOLATILE.search(key)]
    if not hard and not only_left and not only_right:
        print("\nRESULT: identical apart from wall-clock fields.")
        return 0
    print(f"\nRESULT: {len(hard)} substantive differences.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
