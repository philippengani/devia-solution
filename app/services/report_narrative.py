from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from typing import Any, Mapping, Optional

import httpx
from langfuse.openai import OpenAI as LangfuseOpenAI
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import Settings
from app.models.schemas import AnalyzeRequest, ProductObservation, SentimentOutput, TrendOutput

logger = logging.getLogger(__name__)


@dataclass
class NarrativeDraft:
    executive_summary: str
    key_findings: list[str]
    synthesis_mode: str
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


class ReportLLMResponse(BaseModel):
    model_config = ConfigDict(extra='ignore')

    executive_summary: str
    key_findings: list[str] = Field(default_factory=list)


class ReportNarrativeService:
    def __init__(self, settings: Settings, *, langfuse_client=None, openai_client=None) -> None:
        self.settings = settings
        self.langfuse_client = langfuse_client
        self.openai_client = openai_client

    def generate(
        self,
        request: AnalyzeRequest,
        product_data: list[ProductObservation],
        sentiment: SentimentOutput,
        trend: TrendOutput,
        *,
        trace_context: Optional[Mapping[str, str]] = None,
    ) -> NarrativeDraft:
        if self._should_use_openai():
            try:
                return self._generate_with_openai(
                    request,
                    product_data,
                    sentiment,
                    trend,
                    trace_context=trace_context,
                )
            except httpx.HTTPStatusError as exc:
                logger.exception(
                    'OpenAI-compatible synthesis failed with HTTP error for product=%s market=%s model=%s status=%s response=%s',
                    request.product_name,
                    request.market,
                    self.settings.llm_model,
                    exc.response.status_code,
                    self._truncate_text(exc.response.text),
                )
                template_draft = self._generate_with_template(request, product_data, sentiment, trend)
                template_draft.warnings.append(
                    f'LLM synthesis fallback engaged after provider error: {exc}'
                )
                return template_draft
            except Exception as exc:
                logger.exception(
                    'OpenAI-compatible synthesis failed for product=%s market=%s model=%s',
                    request.product_name,
                    request.market,
                    self.settings.llm_model,
                )
                template_draft = self._generate_with_template(request, product_data, sentiment, trend)
                template_draft.warnings.append(
                    f'LLM synthesis fallback engaged after provider error: {exc}'
                )
                return template_draft

        return self._generate_with_template(request, product_data, sentiment, trend)

    def _should_use_openai(self) -> bool:
        return (
            self.settings.report_synthesis_mode == 'openai_compatible'
            and self.settings.llm_enabled
            and bool(self.settings.llm_api_key)
            and self.settings.langfuse_enabled
            and self.langfuse_client is not None
        )

    def _generate_with_template(
        self,
        request: AnalyzeRequest,
        product_data: list[ProductObservation],
        sentiment: SentimentOutput,
        trend: TrendOutput,
    ) -> NarrativeDraft:
        seller_count = len(product_data)
        executive_summary = (
            f'{request.product_name} in {request.market} is trading at an average of '
            f'{trend.avg_price:.2f} {product_data[0].currency} across {seller_count} tracked sellers. '
            f'Pricing is {trend.direction}, with a recommended operating band between '
            f'{trend.recommended_price_floor:.2f} and {trend.recommended_price_ceiling:.2f} '
            f'{product_data[0].currency}. Customer sentiment is {sentiment.label}.'
        )

        key_findings = [
            f'Lowest observed price: {trend.min_price:.2f} {product_data[0].currency}; highest: {trend.max_price:.2f} {product_data[0].currency}.',
            f'Competitive intensity is {trend.competitiveness} and demand signal is {trend.demand_signal}.',
            (
                'Review coverage is insufficient, so the report leans more heavily on pricing signals.'
                if sentiment.label == 'insufficient_data'
                else f'Top review themes: {", ".join(sentiment.key_themes[:3])}.'
            ),
        ]

        return NarrativeDraft(
            executive_summary=executive_summary,
            key_findings=key_findings,
            synthesis_mode='template',
            details={'provider': 'template'},
        )

    def _generate_with_openai(
        self,
        request: AnalyzeRequest,
        product_data: list[ProductObservation],
        sentiment: SentimentOutput,
        trend: TrendOutput,
        *,
        trace_context: Optional[Mapping[str, str]] = None,
    ) -> NarrativeDraft:
        prompt = self.langfuse_client.get_prompt(
            self.settings.report_prompt_name,
            label=self.settings.report_prompt_label,
            type='chat',
            fetch_timeout_seconds=max(1, int(self.settings.request_timeout_seconds)),
        )
        compiled_prompt = prompt.compile(
            request_json=json.dumps(request.model_dump(), ensure_ascii=True),
            product_data_json=json.dumps([item.model_dump() for item in product_data], ensure_ascii=True),
            sentiment_json=json.dumps(sentiment.model_dump(), ensure_ascii=True),
            trend_json=json.dumps(trend.model_dump(), ensure_ascii=True),
        )
        response = self._get_openai_client().chat.completions.create(
            model=self.settings.llm_model,
            temperature=0.2,
            response_format={'type': 'json_object'},
            messages=compiled_prompt,
            name='report-generation',
            metadata={
                'langfuse_prompt_name': prompt.name,
                'product_name': request.product_name,
                'market': request.market,
            },
            langfuse_prompt=prompt,
            trace_id=trace_context.get('trace_id') if trace_context else None,
            parent_observation_id=trace_context.get('parent_span_id') if trace_context else None,
        )

        content = response.choices[0].message.content or '{}'
        parsed = ReportLLMResponse.model_validate(json.loads(content))
        return NarrativeDraft(
            executive_summary=parsed.executive_summary,
            key_findings=list(parsed.key_findings[:3]),
            synthesis_mode='openai_compatible',
            details={
                'provider': 'langfuse_openai',
                'prompt_name': prompt.name,
                'prompt_label': self.settings.report_prompt_label,
                'prompt_version': prompt.version,
                'prompt_labels': list(prompt.labels),
            },
        )

    def _truncate_text(self, value: str, limit: int = 500) -> str:
        if len(value) <= limit:
            return value
        return f'{value[:limit]}...'

    def _get_openai_client(self):
        if self.openai_client is None:
            self.openai_client = LangfuseOpenAI(
                api_key=self.settings.llm_api_key,
                base_url=self.settings.llm_base_url,
                timeout=self.settings.request_timeout_seconds,
            )
        return self.openai_client
