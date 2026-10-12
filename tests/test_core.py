from pathlib import Path

import pytest

from tool_eval.exceptions import InvalidInputError, PromptNotFoundError, WorkspaceError
from tool_eval.prompts import load_prompt
from tool_eval.services.arxiv import validate_arxiv_id
from tool_eval.services.formatting import format_apa_citation, format_markdown_table
from tool_eval.utils.paths import require_text_suffix, resolve_in_workspace, to_ref
from tool_eval.utils.text import append_text, read_text, split_text


def test_resolve_accepts_nested_relative_path(tmp_path: Path) -> None:
    resolved = resolve_in_workspace("papers/a.pdf", workspace=tmp_path)
    assert resolved == tmp_path.resolve() / "papers" / "a.pdf"
    assert to_ref(resolved, workspace=tmp_path) == "papers/a.pdf"


@pytest.mark.parametrize("ref", ["../outside.txt", "a/../../outside.txt", "/etc/passwd", "", " "])
def test_resolve_rejects_escape_and_absolute(tmp_path: Path, ref: str) -> None:
    with pytest.raises(WorkspaceError):
        resolve_in_workspace(ref, workspace=tmp_path)


def test_resolve_rejects_symlink_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "link").symlink_to(tmp_path)
    with pytest.raises(WorkspaceError):
        resolve_in_workspace("link/secret.txt", workspace=workspace)


def test_require_text_suffix() -> None:
    require_text_suffix(Path("review.md"))
    with pytest.raises(WorkspaceError):
        require_text_suffix(Path("script.py"))


def test_append_and_read_respect_size_limit(tmp_path: Path) -> None:
    target = tmp_path / "out" / "note.md"
    append_text(target, "hello", max_bytes=10)
    assert read_text(target, max_bytes=10) == "hello"
    with pytest.raises(WorkspaceError):
        append_text(target, "too long", max_bytes=10)


def test_split_text_covers_input_and_respects_chunk_size() -> None:
    text = " ".join(f"word{i}" for i in range(500))
    chunks = split_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(chunk) <= 300 for chunk in chunks)
    assert chunks[0].startswith("word0")
    assert chunks[-1].endswith("word499")


def test_split_text_rejects_bad_overlap() -> None:
    with pytest.raises(InvalidInputError):
        split_text("abc", chunk_size=10, overlap=10)


def test_apa_citation_formats_authors() -> None:
    one = format_apa_citation(["Ada Lovelace"], year=2024, title="Notes", arxiv_id="2401.00001")
    assert one == "Lovelace, A. (2024). Notes. arXiv. https://arxiv.org/abs/2401.00001"
    two = format_apa_citation(
        ["Ada Lovelace", "Alan Mathison Turing"], year=2024, title="T", arxiv_id="2401.00001"
    )
    assert two.startswith("Lovelace, A., & Turing, A. M. (2024).")


def test_markdown_table_escapes_pipes_and_checks_width() -> None:
    table = format_markdown_table(["a", "b"], [["x|y", "z"]])
    assert table.splitlines() == ["| a | b |", "| --- | --- |", "| x\\|y | z |"]
    with pytest.raises(InvalidInputError):
        format_markdown_table(["a", "b"], [["only one"]])


@pytest.mark.parametrize("value", ["2401.00001", "2401.00001v2", "hep-th/9901001"])
def test_arxiv_id_accepts_valid(value: str) -> None:
    assert validate_arxiv_id(value) == value


@pytest.mark.parametrize("value", ["../../etc", "https://arxiv.org/abs/1", "24.1", ""])
def test_arxiv_id_rejects_invalid(value: str) -> None:
    with pytest.raises(InvalidInputError):
        validate_arxiv_id(value)


def test_prompt_loading_hash_and_render() -> None:
    prompt = load_prompt("extract_keywords", "v1")
    assert len(prompt.sha256) == 64
    assert "7" in prompt.render(max_keywords="7")


@pytest.mark.parametrize(
    ("name", "version"), [("system", "v999"), ("..", "v1"), ("system", "../v1")]
)
def test_prompt_missing_or_unsafe(name: str, version: str) -> None:
    with pytest.raises(PromptNotFoundError):
        load_prompt(name, version)
