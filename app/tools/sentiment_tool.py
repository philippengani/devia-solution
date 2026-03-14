from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from typing import Any, Mapping, Optional

from langfuse.openai import OpenAI as LangfuseOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.config import Settings, get_settings
from app.models.schemas import SentimentOutput

logger = logging.getLogger(__name__)


class SentimentLLMResponse(BaseModel):
    model_config = ConfigDict(extra='ignore')

    label: str
    score: Optional[float]
    positive_signals: list[str] = Field(default_factory=list)
    negative_signals: list[str] = Field(default_factory=list)
    key_themes: list[str] = Field(default_factory=list)


@dataclass
class SentimentToolResult:
    output: SentimentOutput
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SentimentTraceContext:
    trace_id: Optional[str] = None
    parent_observation_id: Optional[str] = None


class HeuristicSentimentAnalyzer:
    POSITIVE_WORDS = {'great', 'excellent', 'fast', 'love', 'good', 'premium', 'reliable', 'quality'}
    NEGATIVE_WORDS = {'bad', 'slow', 'expensive', 'poor', 'hate', 'fragile', 'delay', 'late'}
    THEME_RULES = {
        'delivery': {'fast', 'slow', 'delay', 'late'},
        'price perception': {'expensive'},
        'product quality': {'great', 'excellent', 'good', 'premium', 'reliable', 'quality', 'poor', 'fragile'},
        'brand affinity': {'love', 'hate'},
    }

    def run(self, reviews: Optional[list[str]]) -> SentimentOutput:
        if not reviews:
            return SentimentAnalyzerTool.empty_output('No customer reviews supplied.')

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


class LLMSentimentAnalyzer:
    def __init__(self, settings: Settings, langfuse_client=None, openai_client=None) -> None:
        self.settings = settings
        self.langfuse_client = langfuse_client
        self.openai_client = openai_client or LangfuseOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=settings.request_timeout_seconds,
        )

    def run(
        self,
        reviews: list[str],
        *,
        trace_context: Optional[SentimentTraceContext | Mapping[str, str]] = None,
    ) -> SentimentToolResult:
        if self.langfuse_client is None:
            raise RuntimeError('Langfuse client is not configured.')

        prompt = self.langfuse_client.get_prompt(
            self.settings.sentiment_prompt_name,
            label=self.settings.sentiment_prompt_label,
            type='chat',
            fetch_timeout_seconds=max(1, int(self.settings.request_timeout_seconds)),
        )
        compiled_prompt = prompt.compile(
            review_count=len(reviews),
            reviews_json=json.dumps(reviews, ensure_ascii=True),
        )

        response = self.openai_client.chat.completions.create(
            model=self.settings.llm_model,
            messages=compiled_prompt,
            temperature=0.1,
            response_format={'type': 'json_object'},
            name='sentiment-analysis',
            metadata={
                'langfuse_prompt_name': prompt.name,
                'review_count': len(reviews),
            },
            langfuse_prompt=prompt,
            trace_id=self._trace_id(trace_context),
            parent_observation_id=self._parent_observation_id(trace_context),
        )
        content = response.choices[0].message.content or '{}'
        parsed = SentimentLLMResponse.model_validate(json.loads(content))
        output = SentimentOutput(
            label=self._normalize_label(parsed.label),
            score=self._normalize_score(parsed.score),
            review_count=len(reviews),
            positive_signals=self._normalize_list(parsed.positive_signals),
            negative_signals=self._normalize_list(parsed.negative_signals),
            key_themes=self._normalize_themes(parsed.key_themes),
        )
        return SentimentToolResult(
            output=output,
            details={
                'provider': 'langfuse_openai',
                'effective_mode': 'llm',
                'prompt_name': prompt.name,
                'prompt_label': self.settings.sentiment_prompt_label,
                'prompt_version': prompt.version,
                'prompt_labels': list(prompt.labels),
            },
        )

    def _normalize_label(self, value: str) -> str:
        normalized = (value or '').strip().lower()
        if normalized in {'positive', 'mixed', 'negative', 'insufficient_data'}:
            return normalized
        return 'mixed'

    def _normalize_score(self, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return max(0.0, min(1.0, round(float(value), 2)))

    def _normalize_list(self, values: list[str]) -> list[str]:
        return sorted({item.strip() for item in values if isinstance(item, str) and item.strip()})

    def _normalize_themes(self, values: list[str]) -> list[str]:
        normalized = self._normalize_list(values)
        return normalized or ['No dominant theme detected.']

    def _trace_id(self, trace_context: Optional[SentimentTraceContext | Mapping[str, str]]) -> Optional[str]:
        if trace_context is None:
            return None
        if isinstance(trace_context, Mapping):
            return trace_context.get('trace_id')
        return trace_context.trace_id

    def _parent_observation_id(
        self,
        trace_context: Optional[SentimentTraceContext | Mapping[str, str]],
    ) -> Optional[str]:
        if trace_context is None:
            return None
        if isinstance(trace_context, Mapping):
            return trace_context.get('parent_span_id')
        return trace_context.parent_observation_id


class SentimentAnalyzerTool:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        langfuse_client=None,
        openai_client=None,
    ) -> None:
        self.settings = settings or get_settings()
        self.heuristic = HeuristicSentimentAnalyzer()
        self.langfuse_client = langfuse_client
        self.openai_client = openai_client

    @staticmethod
    def empty_output(reason: str) -> SentimentOutput:
        return SentimentOutput(
            label='insufficient_data',
            score=None,
            review_count=0,
            positive_signals=[],
            negative_signals=[],
            key_themes=[reason],
        )

    def run(
        self,
        reviews: Optional[list[str]],
        *,
        trace_context: Optional[SentimentTraceContext | Mapping[str, str]] = None,
    ) -> SentimentToolResult:
        if not reviews:
            return SentimentToolResult(
                output=self.empty_output('No customer reviews supplied.'),
                details={'provider': 'local', 'effective_mode': 'skipped'},
            )

        heuristic_output = self.heuristic.run(reviews)
        if not self._should_use_llm():
            return SentimentToolResult(
                output=heuristic_output,
                details={'provider': 'local', 'effective_mode': 'heuristic'},
            )

        try:
            llm_analyzer = LLMSentimentAnalyzer(
                self.settings,
                langfuse_client=self.langfuse_client,
                openai_client=self.openai_client,
            )
            return llm_analyzer.run(reviews, trace_context=trace_context)
        except (ValidationError, json.JSONDecodeError, Exception) as exc:
            logger.warning('LLM sentiment analysis failed, falling back to heuristic mode: %s', exc)
            warning = f'LLM sentiment fallback engaged: {exc}'
            return SentimentToolResult(
                output=heuristic_output,
                warnings=[warning],
                details={
                    'provider': 'local',
                    'effective_mode': 'heuristic',
                    'fallback_mode': 'heuristic',
                    'fallback_reason': str(exc),
                    'prompt_name': self.settings.sentiment_prompt_name,
                    'prompt_label': self.settings.sentiment_prompt_label,
                },
            )

    def _should_use_llm(self) -> bool:
        return (
            self.settings.sentiment_analysis_mode == 'llm'
            and self.settings.llm_enabled
            and bool(self.settings.llm_api_key)
            and self.settings.langfuse_enabled
            and self.langfuse_client is not None
        )
