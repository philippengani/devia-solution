from app.core.config import get_settings
from app.models.schemas import AnalyzeRequest
from app.services.langgraph_orchestrator import LangGraphMarketAnalysisOrchestrator
from app.services.native_orchestrator import NativeMarketAnalysisOrchestrator


def test_native_orchestrator_builds_full_response():
    orchestrator = NativeMarketAnalysisOrchestrator()
    response = orchestrator.run(AnalyzeRequest(product_name='PlayStation 5'))
    assert response.report.executive_summary
    assert response.metadata['orchestration_mode'] == 'native'
    assert response.trace[-1] == 'generate_report'
    assert response.trace[2] == 'skip_sentiment'
    assert response.sentiment.label == 'insufficient_data'
    assert response.tool_runs[-1].details['synthesis_mode'] == 'template'
    assert response.metadata['sentiment_analysis_mode'] == 'skipped'


def test_langgraph_orchestrator_builds_full_response():
    orchestrator = LangGraphMarketAnalysisOrchestrator()
    response = orchestrator.run(AnalyzeRequest(product_name='iPhone 15'))
    assert response.report.recommendations
    assert response.metadata['orchestration_mode'] == 'langgraph'
    assert response.trace == [
        'plan_analysis',
        'collect_product_data',
        'skip_sentiment',
        'analyze_trend',
        'generate_report',
    ]
    assert response.plan.selected_competitors
    assert response.tool_runs[1].status == 'skipped'
    assert response.metadata['sentiment_analysis_mode'] == 'skipped'


def test_langgraph_orchestrator_runs_sentiment_when_reviews_are_present():
    orchestrator = LangGraphMarketAnalysisOrchestrator()
    response = orchestrator.run(
        AnalyzeRequest(
            product_name='iPhone 15',
            customer_reviews=['Great product', 'Premium but expensive'],
        )
    )
    assert response.trace == [
        'plan_analysis',
        'collect_product_data',
        'analyze_sentiment',
        'analyze_trend',
        'generate_report',
    ]
    assert response.sentiment.label in {'positive', 'mixed'}
    assert len(response.tool_runs) == 4
    assert response.metadata['sentiment_analysis_mode'] == 'heuristic'


def test_native_orchestrator_records_llm_sentiment_fallback(monkeypatch):
    monkeypatch.setenv('LLM_ENABLED', 'true')
    monkeypatch.setenv('LLM_API_KEY', 'sk-test')
    monkeypatch.setenv('SENTIMENT_ANALYSIS_MODE', 'llm')
    monkeypatch.setenv('LANGFUSE_PUBLIC_KEY', 'pk-lf-test')
    monkeypatch.setenv('LANGFUSE_SECRET_KEY', 'sk-lf-secret')
    get_settings.cache_clear()

    orchestrator = NativeMarketAnalysisOrchestrator(settings=get_settings())
    orchestrator.observability._client = None

    class BrokenLangfuseClient:
        def get_prompt(self, *args, **kwargs):
            raise RuntimeError('prompt unavailable')

    orchestrator.sentiment_tool.langfuse_client = BrokenLangfuseClient()
    response = orchestrator.run(
        AnalyzeRequest(
            product_name='iPhone 15',
            customer_reviews=['Great product', 'Too expensive'],
        )
    )

    sentiment_record = next(item for item in response.tool_runs if item.tool_name == 'sentiment_tool')
    assert sentiment_record.details['fallback_mode'] == 'heuristic'
    assert response.metadata['sentiment_analysis_mode'] == 'heuristic'
    assert response.warnings
