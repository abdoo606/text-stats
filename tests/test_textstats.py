from __future__ import annotations

import pytest

from textstats.analysis import (
    flesch_reading_ease,
    keyword_score,
    keywords,
    reading_level,
    sentiment,
    sentences,
    summary,
    words,
)

SIMPLE = "The cat sat on the mat. The dog ran fast. It was a good day."
HARD = (
    "Notwithstanding the aforementioned considerations, the implementation "
    "of comprehensive institutional methodologies necessitates considerable "
    "administrative deliberation."
)


def test_words_and_sentences():
    assert words("Hello, world!") == ["Hello", "world"]
    assert sentences("One. Two? Three!") == ["One", "Two", "Three"]


def test_arabic_is_tokenised():
    assert words("مرحبا بالعالم") == ["مرحبا", "بالعالم"]


def test_arabic_sentence_split():
    assert sentences("كيف حالك؟ أنا بخير.") == ["كيف حالك", "أنا بخير"]


def test_easy_text_scores_higher_than_dense_text():
    assert flesch_reading_ease(SIMPLE) > flesch_reading_ease(HARD)


def test_reading_ease_is_bounded():
    assert 0 <= flesch_reading_ease(HARD) <= 100


def test_empty_text_is_safe():
    assert flesch_reading_ease("") == 0.0
    assert sentences("") == []


def test_reading_levels():
    assert reading_level(85) == "very easy"
    assert reading_level(65) == "plain"
    assert reading_level(45) == "fairly difficult"
    assert reading_level(10) == "difficult"


def test_keywords_drop_stopwords():
    top = dict(keywords("the cat and the mat and the cat"))
    assert "the" not in top
    assert top["cat"] == 2


def test_keyword_scores_are_descending():
    scores = keyword_score("python python data science python data")
    assert [s for _, s in scores] == sorted([s for _, s in scores], reverse=True)


def test_sentiment_positive_negative_neutral():
    assert sentiment("great excellent amazing")["label"] == "positive"
    assert sentiment("terrible awful crash")["label"] == "negative"
    assert sentiment("the cat sat")["label"] == "neutral"


def test_polarity_bounds():
    result = sentiment("good bad")
    assert -1 <= result["polarity"] <= 1


def test_summary_shape():
    result = summary(SIMPLE)
    assert result["words"] > 0
    assert result["unique_ratio"] <= 1
    assert isinstance(result["top_keywords"], list)
    assert result["reading_level"] in {"very easy", "plain", "fairly difficult", "difficult"}


def test_unique_ratio_of_repeated_text():
    assert summary("cat cat cat")["unique_ratio"] == pytest.approx(0.333, abs=0.01)
