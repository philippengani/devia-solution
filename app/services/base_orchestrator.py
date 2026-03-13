from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from time import perf_counter
from typing import Optional
from uuid import uuid4

from app.core.config import Settings, get_settings
from app.core.errors import ToolExecutionError
from app.models.schemas import (
    AnalysisPlan,
    AnalyzeRequest,
    AnalyzeResponse,
    ReportOutput,
    SentimentOutput,
    ToolExecution,
    TrendOutput,
)
from app.services.report_narrative import ReportNarrativeService
from app.tools.product_data_tool import ProductDataTool
from app.tools.report_tool import ReportGeneratorTool
from app.tools.sentiment_tool import SentimentAnalyzerTool
from app.tools.trend_tool import MarketTrendAnalyzerTool


class BaseMarketAnalysisOrchestrator:
    orchestration_mode = 'langgraph'

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.product_tool = ProductDataTool()
        self.sentiment_tool = SentimentAnalyzerTool()
        self.trend_tool = MarketTrendAnalyzerTool()
        self.report_tool = ReportGeneratorTool(ReportNarrativeService(self.settings))

    def _build_analysis_id(self, request: AnalyzeRequest) -> str:
        request_fingerprint = sha256(
            f'{request.product_name}|{request.market}|{request.competitors}|{request.customer_reviews}'.encode('utf-8')
        ).hexdigest()[:8]
        return f'analysis-{request_fingerprint}-{uuid4().hex[:6]}'

    def _build_plan(self, request: AnalyzeRequest) -> tuple[AnalysisPlan, list[str]]:
        supported_markets = ProductDataTool.supported_markets()
        selected_competitors = (
            request.competitors[:3] or ProductDataTool.default_competitors(request.product_name)
        )
        warnings: list[str] = []
        assumptions = [
            'Market and competitor data are mocked but deterministic so the demo remains reproducible.',
        ]

        if request.market not in supported_markets:
            warnings.append(
                f'Market "{request.market}" is not in the mocked currency map; prices default to USD-style formatting.'
            )
        if not request.customer_reviews:
            warnings.append(
                'No customer reviews were provided, so the workflow will skip direct sentiment analysis.'
            )

        plan = AnalysisPlan(
            objective=f'Produce a market snapshot for {request.product_name} in {request.market}.',
            selected_competitors=selected_competitors,
            requires_sentiment=bool(request.customer_reviews),
            seller_selection_reason=(
                'Competitors supplied by the API caller.'
                if request.competitors
                else 'Default seller set selected from the mocked market catalog.'
            ),
            steps=[
                'plan_analysis',
                'collect_product_data',
                'analyze_sentiment' if request.customer_reviews else 'skip_sentiment',
                'analyze_trend',
                'generate_report',
            ],
            assumptions=assumptions,
        )
        return plan, warnings

    def _run_tool(
        self,
        *,
        step_name: str,
        tool_name: str,
        run_callable,
        details: Optional[dict] = None,
    ):
        started_at = datetime.now(timezone.utc)
        started = perf_counter()

        try:
            result = run_callable()
        except Exception as exc:
            raise ToolExecutionError(tool_name, str(exc)) from exc

        ended_at = datetime.now(timezone.utc)
        duration_ms = int((perf_counter() - started) * 1000)
        record = ToolExecution(
            step_name=step_name,
            tool_name=tool_name,
            status='success',
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            details=details or {},
        )
        return result, record

    def _skipped_tool_record(
        self,
        *,
        step_name: str,
        tool_name: str,
        reason: str,
        details: Optional[dict] = None,
    ) -> ToolExecution:
        timestamp = datetime.now(timezone.utc)
        return ToolExecution(
            step_name=step_name,
            tool_name=tool_name,
            status='skipped',
            started_at=timestamp,
            ended_at=timestamp,
            duration_ms=0,
            details=details or {},
            error_message=reason,
        )

    def _build_response(
        self,
        *,
        analysis_id: str,
        request: AnalyzeRequest,
        plan: AnalysisPlan,
        sentiment: SentimentOutput,
        trend: TrendOutput,
        report: ReportOutput,
        product_data,
        tool_runs: list[ToolExecution],
        trace: list[str],
        warnings: list[str],
    ) -> AnalyzeResponse:
        deduplicated_warnings = list(dict.fromkeys(warnings))
        generated_at = datetime.now(timezone.utc)
        return AnalyzeResponse(
            analysis_id=analysis_id,
            generated_at=generated_at,
            request=request,
            plan=plan,
            product_data=product_data,
            sentiment=sentiment,
            trend=trend,
            report=report,
            tool_runs=tool_runs,
            trace=trace,
            warnings=deduplicated_warnings,
            metadata={
                'orchestration_mode': self.orchestration_mode,
                'report_synthesis_mode': report.synthesis_mode,
                'environment': self.settings.environment,
                'thread_id': analysis_id,
            },
        )
