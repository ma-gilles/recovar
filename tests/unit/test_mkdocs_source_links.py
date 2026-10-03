from types import SimpleNamespace

import pytest

from scripts.mkdocs_source_links import on_page_markdown

pytestmark = pytest.mark.unit


@pytest.fixture
def render(tmp_path):
    docs = tmp_path / "docs"
    (docs / "guide").mkdir(parents=True)
    (docs / "index.md").touch()
    (tmp_path / "source.py").touch()
    (tmp_path / "with space.py").touch()
    page = SimpleNamespace(file=SimpleNamespace(abs_src_path=str(docs / "guide" / "page.md")))
    config = {"docs_dir": str(docs), "repo_url": "https://github.com/ma-gilles/recovar"}
    return lambda text: on_page_markdown(text, page=page, config=config, files=None)


def test_source_link_keeps_label_query_and_fragment(render):
    assert render("See [`source.py`](../../source.py?plain=1#L5).") == (
        "See [`source.py`](https://github.com/ma-gilles/recovar/blob/dev/source.py?plain=1#L5)."
    )


def test_source_link_preserves_url_encoding(render):
    assert render("[source](../../with%20space.py)") == (
        "[source](https://github.com/ma-gilles/recovar/blob/dev/with%20space.py)"
    )


@pytest.mark.parametrize(
    "text",
    [
        "[home](../index.md)",
        "[missing](../../missing.py)",
        "[external](https://example.com/source.py)",
        "[section](#source)",
        "[absolute](/source.py)",
        r"\[example](../../source.py)",
        "![image](../../source.py)",
        "`[example](../../source.py)`",
        "```markdown\n[source](../../source.py)\n```\n",
        "~~~markdown\n[source](../../source.py)\n~~~\n",
        "```markdown\n```not-a-closing-fence\n[source](../../source.py)\n```\n",
    ],
)
def test_document_links_code_and_missing_targets_are_unchanged(render, text):
    assert render(text) == text


def test_source_links_resume_after_a_code_fence(render):
    text = "````markdown\n```\n[example](../../source.py)\n````\n[source](../../source.py)\n"
    result = render(text)
    assert "[example](../../source.py)" in result
    assert result.endswith("[source](https://github.com/ma-gilles/recovar/blob/dev/source.py)\n")
