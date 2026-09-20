from src.preprocessing import remove_html_tags, cleanup_whitespace

def test_remove_html_tags():
    dirty = "<p>Hello <b>World</b></p>"
    assert remove_html_tags(dirty) == "Hello World"

def test_cleanup_whitespace():
    messy = "Hello    World\n\tfrom Python"
    assert cleanup_whitespace(messy) == "Hello World from Python"