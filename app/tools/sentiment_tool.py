from __future__ import annotations

from typing import Optional

from app.models.schemas import SentimentOutput


class SentimentAnalyzerTool:
    POSITIVE_WORDS = {'great', 'excellent', 'fast', 'love', 'good', 'premium', 'reliable', 'quality'}
    NEGATIVE_WORDS = {'bad', 'slow', 'expensive', 'poor', 'hate', 'fragile', 'delay', 'late'}
    THEME_RULES = {
        'delivery': {'fast', 'slow', 'delay', 'late'},
        'price perception': {'expensive'},
        'product quality': {'great', 'excellent', 'good', 'premium', 'reliable', 'quality', 'poor', 'fragile'},
        'brand affinity': {'love', 'hate'},
    }

    def empty_output(self, reason: str) -> SentimentOutput:
        return SentimentOutput(
            label='insufficient_data',
            score=None,
            review_count=0,
            positive_signals=[],
            negative_signals=[],
            key_themes=[reason],
        )

    def run(self, reviews: Optional[list[str]]) -> SentimentOutput:
        if not reviews:
            return self.empty_output('No customer reviews supplied.')

        text = ' '.join(reviews).lower()
        positive = sorted([word for word in self.POSITIVE_WORDS if word in text])
        negative = sorted([word for word in self.NEGATIVE_WORDS if word in text])

        theme_hits: list[str] = []
        for theme, theme_words in self.THEME_RULES.items():
            if any(word in text for word in theme_words):
                theme_hits.append(theme)

        score = 0.52 + 0.08 * len(positive) - 0.09 * len(negative)
        score = max(0.0, min(1.0, round(score, 2)))
        if score >= 0.67:
            label = 'positive'
        elif score >= 0.45:
            label = 'mixed'
        else:
            label = 'negative'

        return SentimentOutput(
            label=label,
            score=score,
            review_count=len(reviews),
            positive_signals=positive,
            negative_signals=negative,
            key_themes=theme_hits or ['No dominant theme detected.'],
        )
