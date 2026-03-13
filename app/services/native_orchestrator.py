from __future__ import annotations

from app.models.schemas import AnalyzeRequest, AnalyzeResponse
from app.services.base_orchestrator import BaseMarketAnalysisOrchestrator


class NativeMarketAnalysisOrchestrator(BaseMarketAnalysisOrchestrator):
    orchestration_mode = 'native'

    def run(self, request: AnalyzeRequest) -> AnalyzeResponse:
        analysis_id = self._build_analysis_id(request)
        plan, warnings = self._build_plan(request)

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
            sentiment, record = self._run_tool(
                step_name='analyze_sentiment',
                tool_name='sentiment_tool',
                run_callable=lambda: self.sentiment_tool.run(request.customer_reviews),
                details={'review_count': len(request.customer_reviews or [])},
            )
            trace.append('analyze_sentiment')
            tool_runs.append(record)
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

        report_result, record = self._run_tool(
            step_name='generate_report',
            tool_name='report_tool',
            run_callable=lambda: self.report_tool.run(request, plan, product_data, sentiment, trend),
        )
        report, report_warnings = report_result
        record.details['synthesis_mode'] = report.synthesis_mode
        record.details['recommendation_count'] = len(report.recommendations)
        trace.append('generate_report')
        tool_runs.append(record)

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
