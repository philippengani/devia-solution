import logging

import httpx

from app.core.config import Settings, get_settings
from app.models.schemas import AnalyzeRequest, ProductObservation, SentimentOutput, TrendOutput
from app.services.report_narrative import ReportNarrativeService


def test_settings_use_openai_api_key_fallback(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.delenv('LLM_API_KEY', raising=False)
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')

    settings = get_settings()

    assert settings.llm_api_key == 'sk-test'
    get_settings.cache_clear()


def test_report_narrative_logs_when_openai_call_fails(caplog, monkeypatch):
    settings = Settings(
        app_name='DevIA Market Analysis Agent',
        environment='test',
        orchestration_mode='langgraph',
        report_synthesis_mode='openai_compatible',
        llm_api_key='sk-test',
        llm_base_url='https://api.openai.com/v1',
        llm_model='gpt-4o-mini',
        llm_enabled=True,
        request_timeout_seconds=20.0,
    )
    service = ReportNarrativeService(settings)

    def raise_status_error(*args, **kwargs):
        request = httpx.Request('POST', 'https://api.openai.com/v1/chat/completions')
        response = httpx.Response(429, request=request, text='rate limit')
        raise httpx.HTTPStatusError('too many requests', request=request, response=response)

    monkeypatch.setattr(service, '_generate_with_openai', raise_status_error)

    with caplog.at_level(logging.ERROR):
        draft = service.generate(
            AnalyzeRequest(product_name='iPhone 15', market='CA'),
            [
                ProductObservation(
                    source='Amazon',
                    product_title='Apple iPhone 15 128GB',
                    price=1111.0,
                    currency='CAD',
                    availability='in_stock',
                    rating=4.6,
                    review_count=1360,
                    search_url='https://mock.market.local/amazon/iphone+15',
                )
            ],
            SentimentOutput(
                label='insufficient_data',
                score=None,
                review_count=0,
                positive_signals=[],
                negative_signals=[],
                key_themes=['No customer reviews supplied.'],
            ),
            TrendOutput(
                direction='stable',
                avg_price=1111.0,
                min_price=1111.0,
                max_price=1111.0,
                price_spread=0.0,
                competitiveness='high',
                demand_signal='medium',
                recommended_price_floor=1100.0,
                recommended_price_ceiling=1120.0,
                insight='Test trend insight.',
            ),
        )

    assert draft.synthesis_mode == 'template'
    assert 'OpenAI-compatible synthesis failed with HTTP error' in caplog.text
