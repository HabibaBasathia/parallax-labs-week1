import re
import unicodedata

def remove_html_tags(text: str) -> str:
    """Removes any HTML tags (like <p>, <br>, etc.) from the text."""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def normalize_unicode(text: str) -> str:
    """Normalizes strange hidden unicode characters into standard text."""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def cleanup_whitespace(text: str) -> str:
    """Replaces multiple spaces, newlines, and tabs with a single space."""
    return re.sub(r'\s+', ' ', text).strip()

def clean_text(text: str) -> str:
    """Runs the full cleaning pipeline on a single text string."""
    if not isinstance(text, str):
        return ""
    text = remove_html_tags(text)
    text = normalize_unicode(text)
    text = cleanup_whitespace(text)
    return text