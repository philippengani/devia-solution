from fastapi.testclient import TestClient

from app.main import app

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
