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
