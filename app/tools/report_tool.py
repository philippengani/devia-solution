from __future__ import annotations

from app.models.schemas import (
    AnalysisPlan,
    AnalyzeRequest,
    ProductObservation,
    ReportOutput,
    SentimentOutput,
    TrendOutput,
)
from app.services.report_narrative import ReportNarrativeService


class ReportGeneratorTool:
    def __init__(self, narrative_service: ReportNarrativeService) -> None:
        self.narrative_service = narrative_service
        self.last_run_details: dict[str, object] = {}

    def run(
        self,
        request: AnalyzeRequest,
        plan: AnalysisPlan,
        product_data: list[ProductObservation],
        sentiment: SentimentOutput,
        trend: TrendOutput,
        *,
        trace_context=None,
    ) -> tuple[ReportOutput, list[str]]:
        narrative = self.narrative_service.generate(
            request,
            product_data,
            sentiment,
            trend,
            trace_context=trace_context,
        )
        self.last_run_details = dict(narrative.details)
        recommendations = self._build_recommendations(request, trend, sentiment, product_data)
        price_chart = self._build_price_chart(product_data)
        markdown = self._build_markdown(
            request=request,
            plan=plan,
            narrative=narrative,
            product_data=product_data,
            sentiment=sentiment,
            trend=trend,
            recommendations=recommendations,
            price_chart=price_chart,
        )

        return (
            ReportOutput(
                executive_summary=narrative.executive_summary,
                key_findings=narrative.key_findings,
                recommendations=recommendations,
                price_chart_markdown=price_chart,
                synthesis_mode=narrative.synthesis_mode,
                markdown=markdown,
            ),
            narrative.warnings,
        )

    def _build_recommendations(
        self,
        request: AnalyzeRequest,
        trend: TrendOutput,
        sentiment: SentimentOutput,
        product_data: list[ProductObservation],
    ) -> list[str]:
        if not request.include_recommendations:
            return []

        strongest_seller = max(product_data, key=lambda item: (item.rating, item.review_count))
        recommendations = [
            (
                f'Anchor pricing between {trend.recommended_price_floor:.2f} and '
                f'{trend.recommended_price_ceiling:.2f} {product_data[0].currency} to stay inside the core market band.'
            ),
            f'Benchmark merchandising and trust signals against {strongest_seller.source}, which leads on rating volume.',
        ]

        if sentiment.label == 'positive':
            recommendations.append('Reuse positive review language in paid acquisition and PDP copy to improve conversion.')
        elif sentiment.label == 'mixed':
            recommendations.append('Address the most common negative themes before increasing spend or widening distribution.')
        elif sentiment.label == 'negative':
            recommendations.append('Fix review pain points before scaling; current sentiment will drag conversion efficiency.')
        else:
            recommendations.append('Collect fresh customer reviews to validate positioning before making a larger inventory bet.')

        return recommendations

    def _build_price_chart(self, product_data: list[ProductObservation]) -> str:
        highest_price = max(item.price for item in product_data)
        lines = ['```text']
        for item in product_data:
            bar_length = max(1, int((item.price / highest_price) * 24))
            lines.append(f'{item.source:<12} {"#" * bar_length} {item.price:.2f} {item.currency}')
        lines.append('```')
        return '\n'.join(lines)

    def _build_markdown(
        self,
        *,
        request: AnalyzeRequest,
        plan: AnalysisPlan,
        narrative,
        product_data: list[ProductObservation],
        sentiment: SentimentOutput,
        trend: TrendOutput,
        recommendations: list[str],
        price_chart: str,
    ) -> str:
        findings = '\n'.join(f'- {item}' for item in narrative.key_findings)
        competitor_rows = '\n'.join(
            (
                f'| {row.source} | {row.price:.2f} {row.currency} | {row.availability} | '
                f'{row.rating:.1f} | {row.review_count} |'
            )
            for row in product_data
        )
        recommendation_lines = (
            '\n'.join(f'- {item}' for item in recommendations)
            if recommendations
            else '- Recommendations intentionally omitted from this request.'
        )
        sentiment_score = f'{sentiment.score:.2f}' if sentiment.score is not None else 'n/a'

        return f"""# Market Analysis Report - {request.product_name}

## Executive summary
{narrative.executive_summary}

## Planned workflow
- Objective: {plan.objective}
- Seller selection: {plan.seller_selection_reason}
- Steps: {', '.join(plan.steps)}
- Assumptions: {', '.join(plan.assumptions)}

## Competitive snapshot
| Source | Price | Availability | Rating | Reviews |
|---|---:|---|---:|---:|
{competitor_rows}

## Price visualization
{price_chart}

## Sentiment
- Label: {sentiment.label}
- Score: {sentiment_score}
- Review count: {sentiment.review_count}
- Positive signals: {', '.join(sentiment.positive_signals) or 'none'}
- Negative signals: {', '.join(sentiment.negative_signals) or 'none'}
- Themes: {', '.join(sentiment.key_themes) or 'none'}

## Market trend
- Direction: {trend.direction}
- Competitiveness: {trend.competitiveness}
- Demand signal: {trend.demand_signal}
- Recommended price band: {trend.recommended_price_floor:.2f} to {trend.recommended_price_ceiling:.2f} {product_data[0].currency}
- Insight: {trend.insight}

## Key findings
{findings}

## Recommendations
{recommendation_lines}
"""
