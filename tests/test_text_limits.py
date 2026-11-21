import json
from pathlib import Path

import pytest
from langchain_core.language_models import FakeListChatModel

from tool_eval.config import Settings, use_settings
from tool_eval.exceptions import InvalidInputError
from tool_eval.services import papers
from tool_eval.services.reading import reading_meter
from tool_eval.tools.level1.files import read_file
from tool_eval.utils.text import split_spans, split_text, truncate


def _settings(workspace: Path, **extra: object) -> Settings:
    return Settings(_env_file=None, workspace_dir=workspace, **extra)  # type: ignore[arg-type]


def _paper(workspace: Path, chars: int, stem: str = "2401.12345") -> str:
    words = " ".join(f"w{i:05d}" for i in range(chars // 7 + 1))[:chars]
    path = workspace / "texts" / f"{stem}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(words, encoding="utf-8")
    return f"texts/{stem}.txt"


def test_cut_text_carries_a_marker() -> None:
    assert truncate("a" * 10, max_chars=4) == "aaaa\n\n[cut: showing 4 of 10 characters]"


def test_short_text_has_no_marker() -> None:
    assert truncate("abc", max_chars=10) == "abc"


def test_spans_match_chunks() -> None:
    text = " ".join(f"word{i}" for i in range(400))
    spans = split_spans(text, chunk_size=300, overlap=50)
    assert [text[a:b] for a, b in spans] == split_text(text, chunk_size=300, overlap=50)


def test_read_file_marks_the_cut_and_records_the_read(tmp_path: Path) -> None:
    ref = _paper(tmp_path, 20_000)
    with use_settings(_settings(tmp_path)), reading_meter() as meter:
        text = read_file.invoke({"path": ref})
    assert text.endswith("[cut: showing 6000 of 20000 characters]")
    assert meter.chars_read() == {"2401.12345": 6000}


def test_whole_paper_is_rejected_by_the_chunk_summarizer(tmp_path: Path) -> None:
    ref = _paper(tmp_path, 20_000)
    llm = FakeListChatModel(responses=["never"])
    with pytest.raises(InvalidInputError, match="split_text_into_chunks"):
        papers.summarize_chunk_file(ref, settings=_settings(tmp_path), llm=llm)
    assert llm.i == 0


def test_chunks_are_summarized_and_counted_once(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ref = _paper(tmp_path, 20_000)
    llm = FakeListChatModel(responses=["summary"] * 20)
    with reading_meter() as meter:
        chunks = papers.split_paper_text(ref, settings=settings)
        for chunk in chunks:
            papers.summarize_chunk_file(chunk, settings=settings, llm=llm)
            papers.summarize_chunk_file(chunk, settings=settings, llm=llm)
    index = json.loads((tmp_path / "chunks" / "2401.12345" / "index.json").read_text())
    covered = max(end for _, end in index.values()) - min(start for start, _ in index.values())
    assert len(chunks) == 6
    assert meter.chars_read() == {"2401.12345": covered}
    assert 16_000 < covered <= 18_000


def test_reads_outside_a_meter_are_ignored(tmp_path: Path) -> None:
    ref = _paper(tmp_path, 100)
    with use_settings(_settings(tmp_path)):
        assert read_file.invoke({"path": ref}).startswith("w00000")
