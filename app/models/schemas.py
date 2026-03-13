from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            'example': {
                'product_name': 'iPhone 15',
                'market': 'CA',
                'competitors': ['Amazon', 'Best Buy', 'Walmart'],
                'include_recommendations': True,
                'customer_reviews': [
                    'Great product and very reliable',
                    'Premium feel but expensive',
                    'Fast delivery and good quality',
                ],
            }
        },
    )

    product_name: str = Field(..., min_length=1, examples=['iPhone 15'])
    market: str = Field(default='CA', min_length=2, max_length=3, examples=['CA'])
    competitors: list[str] = Field(default_factory=list, max_length=5)
    include_recommendations: bool = True
    customer_reviews: Optional[list[str]] = Field(default=None, max_length=20)

    @field_validator('product_name', 'market', mode='before')
    @classmethod
    def _strip_text(cls, value: str) -> str:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator('market')
    @classmethod
    def _normalize_market(cls, value: str) -> str:
        normalized = value.upper()
        if len(normalized) < 2:
            raise ValueError('market must contain at least two characters')
        return normalized

    @field_validator('competitors', mode='before')
    @classmethod
    def _normalize_competitors(cls, value: Optional[list[str]]) -> list[str]:
        if not value:
            return []

        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                continue
            cleaned = item.strip()
            if not cleaned:
                continue
            marker = cleaned.casefold()
            if marker in seen:
                continue
            normalized.append(cleaned)
            seen.add(marker)
        return normalized

    @field_validator('customer_reviews', mode='before')
    @classmethod
    def _normalize_reviews(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        if value is None:
            return None

        normalized = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        return normalized or None


class AnalysisPlan(BaseModel):
    objective: str
    selected_competitors: list[str]
    requires_sentiment: bool
    seller_selection_reason: str
    steps: list[str]
    assumptions: list[str] = Field(default_factory=list)


class ProductObservation(BaseModel):
    source: str
    product_title: str
    price: float
    currency: str
    availability: str
    rating: float
    review_count: int
    search_url: str


class SentimentOutput(BaseModel):
    label: Literal['positive', 'mixed', 'negative', 'insufficient_data']
    score: Optional[float]
    review_count: int
    positive_signals: list[str] = Field(default_factory=list)
    negative_signals: list[str] = Field(default_factory=list)
    key_themes: list[str] = Field(default_factory=list)


class TrendOutput(BaseModel):
    direction: Literal['stable', 'volatile']
    avg_price: float
    min_price: float
    max_price: float
    price_spread: float
    competitiveness: Literal['high', 'medium', 'low']
    demand_signal: Literal['high', 'medium', 'low']
    recommended_price_floor: float
    recommended_price_ceiling: float
    insight: str


class ReportOutput(BaseModel):
    executive_summary: str
    key_findings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    price_chart_markdown: str
    synthesis_mode: Literal['template', 'openai_compatible']
    markdown: str


class ToolExecution(BaseModel):
    step_name: str
    tool_name: str
    status: Literal['success', 'skipped', 'failed']
    started_at: datetime
    ended_at: datetime
    duration_ms: int
    details: dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None


class AnalyzeResponse(BaseModel):
    analysis_id: str
    generated_at: datetime
    request: AnalyzeRequest
    plan: AnalysisPlan
    product_data: list[ProductObservation]
    sentiment: SentimentOutput
    trend: TrendOutput
    report: ReportOutput
    tool_runs: list[ToolExecution] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
