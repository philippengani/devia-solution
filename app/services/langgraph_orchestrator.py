from __future__ import annotations

import operator
from datetime import datetime, timezone
from time import perf_counter

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

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
from app.services.base_orchestrator import BaseMarketAnalysisOrchestrator


class GraphState(TypedDict, total=False):
    analysis_id: str
    request: AnalyzeRequest
    plan: AnalysisPlan
    product_data: list
    sentiment: SentimentOutput
    trend: TrendOutput
    report: ReportOutput
    tool_runs: Annotated[list[ToolExecution], operator.add]
    trace: Annotated[list[str], operator.add]
    warnings: Annotated[list[str], operator.add]


class LangGraphMarketAnalysisOrchestrator(BaseMarketAnalysisOrchestrator):
    orchestration_mode = 'langgraph'

    def __init__(self, settings=None) -> None:
        super().__init__(settings)
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(GraphState)
        workflow.add_node('plan_analysis', self._plan_analysis)
        workflow.add_node('collect_product_data', self._collect_product_data)
        workflow.add_node('analyze_sentiment', self._analyze_sentiment)
        workflow.add_node('skip_sentiment', self._skip_sentiment)
        workflow.add_node('analyze_trend', self._analyze_trend)
        workflow.add_node('generate_report', self._generate_report)

        workflow.add_edge(START, 'plan_analysis')
        workflow.add_edge('plan_analysis', 'collect_product_data')
        workflow.add_conditional_edges(
            'collect_product_data',
            self._route_sentiment,
            {
                'analyze_sentiment': 'analyze_sentiment',
                'skip_sentiment': 'skip_sentiment',
            },
        )
        workflow.add_edge('analyze_sentiment', 'analyze_trend')
        workflow.add_edge('skip_sentiment', 'analyze_trend')
        workflow.add_edge('analyze_trend', 'generate_report')
        workflow.add_edge('generate_report', END)
        return workflow.compile(checkpointer=MemorySaver())

    def _plan_analysis(self, state: GraphState) -> GraphState:
        plan, warnings = self._run_plan_step(state['request'])
        return {'plan': plan, 'trace': ['plan_analysis'], 'warnings': warnings}

    def _collect_product_data(self, state: GraphState) -> GraphState:
        product_data, record = self._run_tool(
            step_name='collect_product_data',
            tool_name='product_data_tool',
            run_callable=lambda: self.product_tool.run(state['request'], state['plan']),
            details={'seller_count': len(state['plan'].selected_competitors)},
        )
        return {'product_data': product_data, 'tool_runs': [record], 'trace': ['collect_product_data']}

    def _route_sentiment(self, state: GraphState) -> str:
        if state['plan'].requires_sentiment:
            return 'analyze_sentiment'
        return 'skip_sentiment'

    def _analyze_sentiment(self, state: GraphState) -> GraphState:
        review_count = len(state['request'].customer_reviews or [])
        details = {'review_count': review_count}
        started_at = datetime.now(timezone.utc)
        started = perf_counter()
        with self.observability.start_step(
            step_name='analyze_sentiment',
            input_data=details,
            metadata=details,
        ) as span:
            try:
                result = self.sentiment_tool.run(
                    state['request'].customer_reviews,
                    trace_context=self.observability.trace_context_from_observation(span),
                )
                if span is not None:
                    span.update(output={'label': result.output.label, 'mode': result.details.get('effective_mode')})
            except Exception as exc:
                raise ToolExecutionError('sentiment_tool', str(exc)) from exc

        ended_at = datetime.now(timezone.utc)
        record = ToolExecution(
            step_name='analyze_sentiment',
            tool_name='sentiment_tool',
            status='success',
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=int((perf_counter() - started) * 1000),
            details=self._build_sentiment_tool_details(details, result.details),
        )
        return {
            'sentiment': result.output,
            'tool_runs': [record],
            'trace': ['analyze_sentiment'],
            'warnings': result.warnings,
        }

    def _skip_sentiment(self, state: GraphState) -> GraphState:
        reason = 'No customer reviews supplied.'
        record = self._skipped_tool_record(
            step_name='skip_sentiment',
            tool_name='sentiment_tool',
            reason=reason,
        )
        return {
            'sentiment': self.sentiment_tool.empty_output(reason),
            'tool_runs': [record],
            'trace': ['skip_sentiment'],
            'warnings': [reason],
        }

    def _analyze_trend(self, state: GraphState) -> GraphState:
        trend, record = self._run_tool(
            step_name='analyze_trend',
            tool_name='trend_tool',
            run_callable=lambda: self.trend_tool.run(state['product_data']),
            details={'observation_count': len(state['product_data'])},
        )
        return {'trend': trend, 'tool_runs': [record], 'trace': ['analyze_trend']}

    def _generate_report(self, state: GraphState) -> GraphState:
        details = {
            'observation_count': len(state['product_data']),
            'include_recommendations': state['request'].include_recommendations,
        }
        started_at = datetime.now(timezone.utc)
        started = perf_counter()
        with self.observability.start_step(
            step_name='generate_report',
            input_data=details,
            metadata=details,
        ) as span:
            try:
                report, report_warnings = self.report_tool.run(
                    state['request'],
                    state['plan'],
                    state['product_data'],
                    state['sentiment'],
                    state['trend'],
                    trace_context=self.observability.trace_context_from_observation(span),
                )
                if span is not None:
                    span.update(output={'synthesis_mode': report.synthesis_mode})
            except Exception as exc:
                raise ToolExecutionError('report_tool', str(exc)) from exc
        record = ToolExecution(
            step_name='generate_report',
            tool_name='report_tool',
            status='success',
            started_at=started_at,
            ended_at=datetime.now(timezone.utc),
            duration_ms=int((perf_counter() - started) * 1000),
            details={
                **details,
                **self.report_tool.last_run_details,
                'synthesis_mode': report.synthesis_mode,
                'recommendation_count': len(report.recommendations),
            },
        )
        return {
            'report': report,
            'tool_runs': [record],
            'trace': ['generate_report'],
            'warnings': report_warnings,
        }

    def run(self, request: AnalyzeRequest) -> AnalyzeResponse:
        analysis_id = self._build_analysis_id(request)
        with self.observability.start_request(
            analysis_id=analysis_id,
            request=request,
            orchestration_mode=self.orchestration_mode,
        ) as trace_span:
            self.current_langfuse_trace_id = getattr(trace_span, 'trace_id', None)
            result = self.graph.invoke(
                {
                    'analysis_id': analysis_id,
                    'request': request,
                    'tool_runs': [],
                    'trace': [],
                    'warnings': [],
                },
                config={'configurable': {'thread_id': analysis_id}},
            )
            if trace_span is not None:
                trace_span.update(
                    output={
                        'steps': result.get('trace', []),
                        'report_mode': result['report'].synthesis_mode,
                    }
                )
            self.observability.flush()
        return self._build_response(
            analysis_id=analysis_id,
            request=request,
            plan=result['plan'],
            product_data=result['product_data'],
            sentiment=result['sentiment'],
            trend=result['trend'],
            report=result['report'],
            tool_runs=result.get('tool_runs', []),
            trace=result.get('trace', []),
            warnings=result.get('warnings', []),
        )

    def render_mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()

    def render_ascii(self) -> str:
        try:
            return self.graph.get_graph().draw_ascii()
        except ImportError:
            return 'ASCII rendering unavailable: install grandalf to enable draw_ascii().'
