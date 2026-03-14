from fastapi.testclient import TestClient

from app.main import app
from app.core.config import get_settings

client = TestClient(app)


def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'
    assert response.json()['orchestration_mode'] == 'langgraph'


def test_analyze_endpoint():
    payload = {
        'product_name': 'iPhone 15',
        'market': 'CA',
        'customer_reviews': ['Great device', 'Premium but expensive'],
    }
    response = client.post('/analyze', json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body['analysis_id'].startswith('analysis-')
    assert body['plan']['selected_competitors']
    assert body['report']['executive_summary']
    assert body['trend']['avg_price'] > 0
    assert len(body['tool_runs']) == 4


def test_analyze_endpoint_rejects_blank_product_name():
    response = client.post('/analyze', json={'product_name': '   ', 'market': 'CA'})
    assert response.status_code == 422


def test_workflow_diagram_endpoint():
    response = client.get('/workflow/diagram')
    assert response.status_code == 200
    body = response.json()
    assert body['orchestration_mode'] == 'langgraph'
    assert 'plan_analysis' in body['mermaid']
    assert body['ascii']


def test_analyze_endpoint_llm_fallback_metadata(monkeypatch):
    monkeypatch.setenv('LLM_ENABLED', 'true')
    monkeypatch.setenv('LLM_API_KEY', 'sk-test')
    monkeypatch.setenv('SENTIMENT_ANALYSIS_MODE', 'llm')
    monkeypatch.setenv('LANGFUSE_PUBLIC_KEY', 'pk-lf-test')
    monkeypatch.setenv('LANGFUSE_SECRET_KEY', 'sk-lf-secret')
    get_settings.cache_clear()

    from app.api import routes
    from app.services.native_orchestrator import NativeMarketAnalysisOrchestrator

    def fake_get_orchestrator():
        orchestrator = NativeMarketAnalysisOrchestrator(settings=get_settings())
        orchestrator.observability._client = None

        class BrokenLangfuseClient:
            def get_prompt(self, *args, **kwargs):
                raise RuntimeError('prompt unavailable')

        orchestrator.sentiment_tool.langfuse_client = BrokenLangfuseClient()
        return orchestrator

    monkeypatch.setattr(routes, 'get_orchestrator', fake_get_orchestrator)

    payload = {
        'product_name': 'iPhone 15',
        'market': 'CA',
        'customer_reviews': ['Great device', 'Premium but expensive'],
    }
    response = client.post('/analyze', json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body['metadata']['sentiment_analysis_mode'] == 'heuristic'
    sentiment_record = next(item for item in body['tool_runs'] if item['tool_name'] == 'sentiment_tool')
    assert sentiment_record['details']['fallback_mode'] == 'heuristic'
    assert body['warnings']
