"""
NLP helpers: sentiment (VADER) and topic modeling (LDA).

Sentiment: VADER gives a score from -1 (very negative) to +1 (very positive).
Topics:    LDA groups words that often appear together into "topics".
           Each chunk is then assigned to its best-matching topic.
"""

from typing import List, Tuple

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

MIN_WORDS: int = 5          # chunks shorter than this are too short to label
N_TOPICS: int = 8           # how many topics to discover
POSITIVE_CUTOFF: float = 0.05
NEGATIVE_CUTOFF: float = -0.05

_analyzer = SentimentIntensityAnalyzer()


# ---------- Sentiment ----------

def get_sentiment(text: str) -> str:
    """
    Label the sentiment of a text using VADER.

    Args:
        text: The text to analyze.

    Returns:
        "positive", "negative" or "neutral". Empty or whitespace-only
        text is "neutral".
    """
    if not text or not text.strip():
        return "neutral"  # edge case: empty text

    score = _analyzer.polarity_scores(text)["compound"]

    if score >= POSITIVE_CUTOFF:
        return "positive"
    if score <= NEGATIVE_CUTOFF:
        return "negative"
    return "neutral"


# ---------- Topics ----------

def fit_topic_model(
    texts: List[str], n_topics: int = N_TOPICS
) -> Tuple[CountVectorizer, LatentDirichletAllocation, List[str]]:
    """
    Learn topics from all the chunks with LDA.

    Common words, words in over half the chunks, and rare words (in fewer
    than 5 chunks, such as jargon or typos) are ignored.

    Args:
        texts: All chunk texts to learn from.
        n_topics: How many topics to find.

    Returns:
        A tuple of (vectorizer, lda_model, topic_names). Each topic is
        named by its 3 strongest words, for example "friend_park_play".
    """
    vectorizer = CountVectorizer(
        stop_words="english",  # ignore words like "the", "and"
        max_df=0.5,            # ignore words that appear in over half the chunks
        min_df=5,              # ignore rare words (jargon, typos)
        max_features=5000,
    )
    word_counts = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,       # same result every run
        learning_method="batch",
        max_iter=15,
    )
    lda.fit(word_counts)

    # Name each topic by its 3 strongest words
    words = vectorizer.get_feature_names_out()
    topic_names: List[str] = []
    for topic_weights in lda.components_:
        top_ids = topic_weights.argsort()[-3:][::-1]
        topic_names.append("_".join(words[i] for i in top_ids))

    return vectorizer, lda, topic_names


def assign_topics(
    texts: List[str],
    vectorizer: CountVectorizer,
    lda: LatentDirichletAllocation,
    topic_names: List[str],
) -> List[str]:
    """
    Give each chunk the name of its best-matching topic.

    Args:
        texts: The chunk texts to label.
        vectorizer: The fitted vectorizer from fit_topic_model.
        lda: The fitted LDA model from fit_topic_model.
        topic_names: The topic names from fit_topic_model.

    Returns:
        One topic name per text, in the same order. Chunks that are too
        short (under MIN_WORDS words), or that contain only words the
        model does not know, get "uncategorized".
    """
    word_counts = vectorizer.transform(texts)
    topic_scores = lda.transform(word_counts)

    results: List[str] = []
    for i, text in enumerate(texts):
        too_short = len(text.split()) < MIN_WORDS
        no_known_words = word_counts[i].sum() == 0

        if too_short or no_known_words:
            results.append("uncategorized")
        else:
            best_topic = topic_scores[i].argmax()
            results.append(topic_names[best_topic])

    return results