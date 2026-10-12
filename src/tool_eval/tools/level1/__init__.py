"""Level 1: micro tools, one small operation each."""

from langchain_core.tools import BaseTool

from tool_eval.tools.level1.documents import (
    download_pdf,
    extract_text_from_pdf,
    split_text_into_chunks,
)
from tool_eval.tools.level1.files import append_to_file, read_file
from tool_eval.tools.level1.formatting import create_markdown_table, format_apa_citation
from tool_eval.tools.level1.language import extract_keywords, summarize_chunk, translate_text
from tool_eval.tools.level1.search import search_arxiv, search_wikipedia

TOOLS: list[BaseTool] = [
    search_arxiv,
    search_wikipedia,
    download_pdf,
    extract_text_from_pdf,
    split_text_into_chunks,
    summarize_chunk,
    extract_keywords,
    translate_text,
    format_apa_citation,
    create_markdown_table,
    append_to_file,
    read_file,
]
