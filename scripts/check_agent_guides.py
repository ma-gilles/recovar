"""Check mirrored agent contracts and relative file links in development guides."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

MIRRORS = (
    ("AGENTS.md", "CLAUDE.md"),
    ("recovar/em/AGENTS.md", "recovar/em/CLAUDE.md"),
)


def check_guides(root: Path) -> list[str]:
    errors = []
    for first, second in MIRRORS:
        paths = root / first, root / second
        if not all(path.is_file() for path in paths):
            errors.append(f"missing required mirror: {first} or {second}")
        elif paths[0].read_bytes() != paths[1].read_bytes():
            errors.append(f"agent contracts differ: {first} and {second}")

    guides = {root / name for pair in MIRRORS for name in pair}
    guides.update(root / name for name in ("CONTRIBUTING.md", "recovar/CLAUDE.md", "tests/CLAUDE.md"))
    guides.update((root / "docs/development").glob("*.md"))
    for guide in sorted(guides):
        if not guide.is_file():
            errors.append(f"missing guide: {guide.relative_to(root)}")
            continue
        content = guide.read_text()
        if not content.strip():
            errors.append(f"empty guide: {guide.relative_to(root)}")
        fenced = False
        for line_number, line in enumerate(content.splitlines(), 1):
            if line.lstrip().startswith("```"):
                fenced = not fenced
                continue
            if fenced:
                continue
            for target in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", line):
                link = urlsplit(target)
                if link.scheme or link.netloc or not link.path:
                    continue
                path = guide.parent / unquote(link.path)
                if not path.exists():
                    errors.append(f"{guide.relative_to(root)}:{line_number}: missing link target {target}")
    return errors


def main() -> int:
    errors = check_guides(Path(__file__).resolve().parents[1])
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Agent mirrors and development-guide file links are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
