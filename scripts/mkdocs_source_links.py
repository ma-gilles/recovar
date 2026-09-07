"""Keep repository-relative source links usable in the published documentation."""

import re
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit

_LINK_OR_CODE = re.compile(r"`+[^`]*`+|(?<![!\\])\[[^\]\n]*\]\((?P<url>[^\s)]+)\)")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


def on_page_markdown(markdown, *, page, config, files):
    """Link existing files outside docs to the repository's development branch.

    Authored Markdown keeps its local links for repository readers. Missing
    targets remain unchanged so MkDocs still reports them during validation.
    """
    docs = Path(config["docs_dir"]).resolve()
    repo = docs.parent
    source = Path(page.file.abs_src_path).resolve()

    def replace(match):
        target = match.group("url")
        if target is None:
            return match.group(0)
        url = urlsplit(target)
        if url.scheme or url.netloc or not url.path or url.path.startswith("/"):
            return match.group(0)
        path = (source.parent / unquote(url.path)).resolve()
        if not path.is_file() or not path.is_relative_to(repo) or path.is_relative_to(docs):
            return match.group(0)
        href = config["repo_url"].rstrip("/") + "/blob/dev/" + quote(path.relative_to(repo).as_posix())
        href = urlunsplit((*urlsplit(href)[:3], url.query, url.fragment))
        start, end = match.span("url")
        return match.group(0)[: start - match.start()] + href + match.group(0)[end - match.start() :]

    result = []
    fence = None
    for line in markdown.splitlines(keepends=True):
        marker = _FENCE.match(line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence) and not line[marker.end() :].strip():
                fence = None
            result.append(line)
        else:
            result.append(line if fence else _LINK_OR_CODE.sub(replace, line))
    return "".join(result)
