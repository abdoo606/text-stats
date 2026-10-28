# 21 · textstats

Readability, keywords and sentiment for English and Arabic.

## Run

```bash
from textstats.analysis import summary
```

## Test

```bash
pytest projects/21-text-stats -q
```

## What it demonstrates

- Flesch reading ease with a vowel-group syllable heuristic
- Non-Latin scripts fall back to a length-based estimate instead of crashing
- Stopword-aware keyword extraction for both English and Arabic
- Lexicon sentiment with a neutral band instead of a fake score
