"""
Text cleaning functions for the corpus.

Pipeline (see clean_text): strip HTML tags -> normalize Unicode to ASCII
-> collapse whitespace.
"""

import re
import unicodedata


def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags (like <p> or <br>) from the text.

    Args:
        text: Text that may contain HTML tags.

    Returns:
        The text with all tags removed.
    """
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def normalize_unicode(text: str) -> str:
    """
    Convert unusual Unicode characters into plain ASCII text.

    Characters that have no ASCII form (for example non-English letters
    or emoji) are dropped.

    Args:
        text: Text that may contain special Unicode characters.

    Returns:
        ASCII-only text.
    """
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')


def cleanup_whitespace(text: str) -> str:
    """
    Replace runs of spaces, newlines and tabs with a single space.

    Args:
        text: Text with messy whitespace.

    Returns:
        The text with single spaces and no leading or trailing whitespace.
    """
    return re.sub(r'\s+', ' ', text).strip()


def clean_text(text: str) -> str:
    """
    Run the full cleaning pipeline on one text.

    Args:
        text: The raw text. Anything that is not a string is treated as empty.

    Returns:
        The cleaned text, or an empty string if the input was not a string.
    """
    if not isinstance(text, str):
        return ""

    text = remove_html_tags(text)
    text = normalize_unicode(text)
    text = cleanup_whitespace(text)
    return text