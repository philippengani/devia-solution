from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging

import httpx

from app.core.config import Settings
from app.models.schemas import AnalyzeRequest, ProductObservation, SentimentOutput, TrendOutput

logger = logging.getLogger(__name__)


@dataclass
class NarrativeDraft:
    executive_summary: str
    key_findings: list[str]
    synthesis_mode: str
    warnings: list[str] = field(default_factory=list)


class ReportNarrativeService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate(
        self,
        request: AnalyzeRequest,
        product_data: list[ProductObservation],
        sentiment: SentimentOutput,
        trend: TrendOutput,
    ) -> NarrativeDraft:
        if self._should_use_openai():
            try:
                return self._generate_with_openai(request, product_data, sentiment, trend)
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
        )

    def _generate_with_openai(
        self,
        request: AnalyzeRequest,
        product_data: list[ProductObservation],
        sentiment: SentimentOutput,
        trend: TrendOutput,
    ) -> NarrativeDraft:
        payload = {
            'model': self.settings.llm_model,
            'temperature': 0.2,
            'response_format': {'type': 'json_object'},
            'messages': [
                {
                    'role': 'system',
                    'content': (
                        'You are a market-intelligence analyst. '
                        'Return compact JSON with keys executive_summary and key_findings. '
                        'key_findings must be a list of exactly 3 concise bullet strings.'
                    ),
                },
                {
                    'role': 'user',
                    'content': json.dumps(
                        {
                            'request': request.model_dump(),
                            'product_data': [item.model_dump() for item in product_data],
                            'sentiment': sentiment.model_dump(),
                            'trend': trend.model_dump(),
                        }
                    ),
                },
            ],
        }

        headers = {
            'Authorization': f'Bearer {self.settings.llm_api_key}',
            'Content-Type': 'application/json',
        }

        with httpx.Client(timeout=self.settings.request_timeout_seconds) as client:
            response = client.post(
                f'{self.settings.llm_base_url}/chat/completions',
                headers=headers,
                json=payload,
            )
            response.raise_for_status()

        content = response.json()['choices'][0]['message']['content']
        parsed = json.loads(content)
        return NarrativeDraft(
            executive_summary=parsed['executive_summary'],
            key_findings=list(parsed['key_findings']),
            synthesis_mode='openai_compatible',
        )

    def _truncate_text(self, value: str, limit: int = 500) -> str:
        if len(value) <= limit:
            return value
        return f'{value[:limit]}...'
