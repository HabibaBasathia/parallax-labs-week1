import pytest
from src.chunking import recursive_chunk


def test_empty_text_returns_no_chunks():
    assert recursive_chunk("") == []
    assert recursive_chunk("   ") == []


def test_short_text_returns_single_chunk():
    text = "This is a short document."
    result = recursive_chunk(text, chunk_size=500)
    assert result == [text]


def test_long_text_is_split_into_multiple_chunks():
    text = "Sentence one. " * 100
    result = recursive_chunk(text, chunk_size=200, chunk_overlap=20)
    assert len(result) > 1
    for chunk in result:
        assert len(chunk) <= 250


def test_overlap_is_applied_between_chunks():
    text = "AAAAAAAAAA " * 30
    result = recursive_chunk(text, chunk_size=50, chunk_overlap=10)
    assert len(result) > 1
    assert result[0][-5:] in result[1]


def test_paragraph_boundaries_are_respected_when_possible():
    text = "Para one line.\n\nPara two line.\n\nPara three line."
    result = recursive_chunk(text, chunk_size=15, chunk_overlap=0, separators=["\n\n"])
    assert len(result) >= 3