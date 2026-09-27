"""
Recursive text chunking.

Strategy: try to split on the "biggest" separator first (paragraphs),
and only fall back to smaller separators (sentences, then words) for
pieces that are still too large. This keeps semantically related text
together as much as possible, which helps embedding quality later.
"""

from typing import List

DEFAULT_SEPARATORS = ["\n\n", ". ", " "]  # paragraph -> sentence -> word


def recursive_chunk(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: List[str] = None,
) -> List[str]:
    """
    Split `text` into chunks of at most `chunk_size` characters,
    with `chunk_overlap` characters repeated between consecutive chunks
    so context isn't lost at chunk boundaries.
    """
    if not text or not text.strip():
        return []

    separators = separators or DEFAULT_SEPARATORS

    # Base case: text already fits
    if len(text) <= chunk_size:
        return [text.strip()]

    # Pick the first separator that actually splits the text
    separator = separators[0] if separators else None
    remaining_separators = separators[1:] if len(separators) > 1 else []

    if separator:
        pieces = text.split(separator)
    else:
        pieces = list(text)  # last resort: char-level split

    chunks = []
    current = ""

    for piece in pieces:
        candidate = (current + separator + piece) if current else piece

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            # piece itself might still be too big -> recurse with smaller separators
            if len(piece) > chunk_size and remaining_separators:
                chunks.extend(
                    recursive_chunk(piece, chunk_size, chunk_overlap, remaining_separators)
                )
                current = ""
            else:
                current = piece

    if current:
        chunks.append(current.strip())

    # Add overlap between consecutive chunks
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-chunk_overlap:]
            overlapped.append((prev_tail + " " + chunks[i]).strip())
        chunks = overlapped

    return [c for c in chunks if c]