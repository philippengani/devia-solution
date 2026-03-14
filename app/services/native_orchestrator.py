from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter

from app.core.errors import ToolExecutionError
from app.models.schemas import AnalyzeRequest, AnalyzeResponse, ToolExecution
from app.services.base_orchestrator import BaseMarketAnalysisOrchestrator


class NativeMarketAnalysisOrchestrator(BaseMarketAnalysisOrchestrator):
    orchestration_mode = 'native'

    def run(self, request: AnalyzeRequest) -> AnalyzeResponse:
        analysis_id = self._build_analysis_id(request)
        with self.observability.start_request(
            analysis_id=analysis_id,
            request=request,
            orchestration_mode=self.orchestration_mode,
        ) as trace_span:
            self.current_langfuse_trace_id = getattr(trace_span, 'trace_id', None)
            plan, warnings = self._run_plan_step(request)

            trace = ['plan_analysis']
            tool_runs = []

            product_data, record = self._run_tool(
                step_name='collect_product_data',
                tool_name='product_data_tool',
                run_callable=lambda: self.product_tool.run(request, plan),
                details={'seller_count': len(plan.selected_competitors)},
            )
            trace.append('collect_product_data')
            tool_runs.append(record)

            if plan.requires_sentiment:
                review_count = len(request.customer_reviews or [])
                details = {'review_count': review_count}
                started_at = datetime.now(timezone.utc)
                started = perf_counter()
                with self.observability.start_step(
                    step_name='analyze_sentiment',
                    input_data=details,
                    metadata=details,
                ) as span:
                    try:
                        sentiment_result = self.sentiment_tool.run(
                            request.customer_reviews,
                            trace_context=self.observability.trace_context_from_observation(span),
                        )
                        if span is not None:
                            span.update(
                                output={
                                    'label': sentiment_result.output.label,
                                    'mode': sentiment_result.details.get('effective_mode'),
                                }
                            )
                    except Exception as exc:
                        raise ToolExecutionError('sentiment_tool', str(exc)) from exc
                trace.append('analyze_sentiment')
                sentiment = sentiment_result.output
                warnings.extend(sentiment_result.warnings)
                tool_runs.append(
                    ToolExecution(
                        step_name='analyze_sentiment',
                        tool_name='sentiment_tool',
                        status='success',
                        started_at=started_at,
                        ended_at=datetime.now(timezone.utc),
                        duration_ms=int((perf_counter() - started) * 1000),
                        details=self._build_sentiment_tool_details(details, sentiment_result.details),
                    )
                )
            else:
                trace.append('skip_sentiment')
                sentiment = self.sentiment_tool.empty_output('No customer reviews supplied.')
                tool_runs.append(
                    self._skipped_tool_record(
                        step_name='skip_sentiment',
                        tool_name='sentiment_tool',
                        reason='No customer reviews supplied.',
                    )
                )

            trend, record = self._run_tool(
                step_name='analyze_trend',
                tool_name='trend_tool',
                run_callable=lambda: self.trend_tool.run(product_data),
                details={'observation_count': len(product_data)},
            )
            trace.append('analyze_trend')
            tool_runs.append(record)

            details = {
                'observation_count': len(product_data),
                'include_recommendations': request.include_recommendations,
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
                        request,
                        plan,
                        product_data,
                        sentiment,
                        trend,
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
            trace.append('generate_report')
            tool_runs.append(record)

            if trace_span is not None:
                trace_span.update(output={'steps': trace, 'report_mode': report.synthesis_mode})
            self.observability.flush()

            return self._build_response(
                analysis_id=analysis_id,
                request=request,
                plan=plan,
                product_data=product_data,
                sentiment=sentiment,
                trend=trend,
                report=report,
                tool_runs=tool_runs,
                trace=trace,
                warnings=[*warnings, *report_warnings],
            )
