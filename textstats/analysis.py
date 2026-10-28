"""Text metrics that work on English and Arabic without any model download."""

from __future__ import annotations

import math
import re
from collections import Counter

WORD = re.compile(r"[\w\u0600-\u06FF']+")
SENTENCE = re.compile(r"[.!?؟।\n]+")
STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for from
    with without by as is are was were be been being it its it's i you he she we they
    not no do does did have has had will would can could should about into over under
    من في على إلى عن مع هذا هذه ذلك التي الذي كان كانت هو هي أن إن لا ما لم قد
    """.split()
)
POSITIVE = frozenset("good great excellent amazing love best happy win fast easy helpful".split())
NEGATIVE = frozenset("bad terrible awful hate worst slow fail broken bug crash poor difficult".split())


def words(text: str) -> list[str]:
    return WORD.findall(text)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE.split(text) if s.strip()]


def _count_syllables(word: str) -> int:
    """Vowel-group heuristic; non-Latin scripts fall back to a length-based estimate."""
    if not re.search(r"[a-zA-Z]", word):
        return max(1, len(word) // 3)
    groups = re.findall(r"[aeiouy]+", word.lower())
    return max(1, len(groups))


def flesch_reading_ease(text: str) -> float:
    word_list = words(text)
    sentence_list = sentences(text)
    if not word_list or not sentence_list:
        return 0.0
    syllables = sum(_count_syllables(w) for w in word_list)
    score = 206.835 - 1.015 * (len(word_list) / len(sentence_list)) - 84.6 * (syllables / len(word_list))
    return round(max(0.0, min(100.0, score)), 1)


def reading_level(score: float) -> str:
    if score >= 80:
        return "very easy"
    if score >= 60:
        return "plain"
    if score >= 40:
        return "fairly difficult"
    return "difficult"


def keywords(text: str, *, top: int = 10) -> list[tuple[str, int]]:
    """Frequency minus stopwords. Case-folded, keeps Arabic intact."""
    counts = Counter(w.lower() for w in words(text) if w.lower() not in STOPWORDS and len(w) > 1)
    return counts.most_common(top)


def keyword_score(text: str, *, top: int = 10) -> list[tuple[str, float]]:
    """TF-IDF-shaped score against the document itself: rare terms win."""
    tokens = [w.lower() for w in words(text)]
    counts = Counter(t for t in tokens if t not in STOPWORDS)
    total = sum(counts.values()) or 1
    doc_sentences = [set(words(s.lower())) for s in sentences(text)] or [set()]
    scored = []
    for term, freq in counts.items():
        containing = sum(term in s for s in doc_sentences) or 1
        idf = math.log((len(doc_sentences) + 1) / containing) + 1
        scored.append((term, round(freq / total * idf, 4)))
    return sorted(scored, key=lambda kv: -kv[1])[:top]


def sentiment(text: str) -> dict[str, object]:
    tokens = [w.lower() for w in words(text)]
    pos = sum(t in POSITIVE for t in tokens)
    neg = sum(t in NEGATIVE for t in tokens)
    total = pos + neg
    polarity = (pos - neg) / total if total else 0.0
    label = "neutral" if abs(polarity) < 0.2 else ("positive" if polarity > 0 else "negative")
    return {"positive": pos, "negative": neg, "polarity": round(polarity, 3), "label": label}


def summary(text: str) -> dict[str, object]:
    word_list = words(text)
    sentence_list = sentences(text)
    score = flesch_reading_ease(text)
    return {
        "characters": len(text),
        "words": len(word_list),
        "sentences": len(sentence_list),
        "avg_words_per_sentence": round(len(word_list) / len(sentence_list), 1) if sentence_list else 0.0,
        "unique_ratio": round(len(set(w.lower() for w in word_list)) / len(word_list), 3) if word_list else 0.0,
        "reading_ease": score,
        "reading_level": reading_level(score),
        "sentiment": sentiment(text)["label"],
        "top_keywords": keywords(text, top=5),
    }
