from src.nlp_analysis import (
    get_sentiment,
    fit_topic_model,
    assign_topics,
)

# A tiny made-up corpus: 3 themes, 10 copies each (30 texts).
# Each word appears in 10 of 30 texts, so it passes the min_df / max_df rules.
SAMPLE_TEXTS = (
    ["the dog barked at the cat near the farm"] * 10
    + ["she baked bread and cake in the kitchen oven"] * 10
    + ["the boat sailed across the ocean with the waves"] * 10
)


def build_model():
    return fit_topic_model(SAMPLE_TEXTS, n_topics=3)


# ---------- Sentiment tests ----------

def test_happy_text_is_positive():
    assert get_sentiment("I love this happy wonderful day") == "positive"


def test_sad_text_is_negative():
    assert get_sentiment("This is terrible, I hate it and I feel so sad") == "negative"


def test_empty_text_is_neutral():
    assert get_sentiment("") == "neutral"
    assert get_sentiment("   ") == "neutral"


# ---------- Topic tests ----------

def test_model_creates_requested_number_of_topics():
    vectorizer, lda, topic_names = build_model()
    assert len(topic_names) == 3


def test_very_short_chunk_is_uncategorized():
    vectorizer, lda, topic_names = build_model()
    result = assign_topics(["Hi there"], vectorizer, lda, topic_names)
    assert result == ["uncategorized"]


def test_unknown_words_are_uncategorized():
    vectorizer, lda, topic_names = build_model()
    jargon = "zzzxqv blorptastic quibbleflux gromnix snarfle"
    result = assign_topics([jargon], vectorizer, lda, topic_names)
    assert result == ["uncategorized"]


def test_normal_chunk_gets_a_real_topic():
    vectorizer, lda, topic_names = build_model()
    text = "the dog barked at the cat near the farm"
    result = assign_topics([text], vectorizer, lda, topic_names)
    assert result[0] in topic_names
    assert result[0] != "uncategorized"