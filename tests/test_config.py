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
        sentiment_analysis_mode='heuristic',
        llm_api_key='sk-test',
        llm_base_url='https://api.openai.com/v1',
        llm_model='gpt-4o-mini',
        llm_enabled=True,
        langfuse_public_key='pk-lf-test',
        langfuse_secret_key='sk-lf-secret',
        langfuse_base_url='https://us.cloud.langfuse.com',
        langfuse_enabled=True,
        sentiment_prompt_name='sentiment-analyzer',
        sentiment_prompt_label='production',
        report_prompt_name='market-analysis-report-generator',
        report_prompt_label='production',
        request_timeout_seconds=20.0,
    )
    service = ReportNarrativeService(settings, langfuse_client=object(), openai_client=object())

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


def test_report_narrative_uses_langfuse_prompt(monkeypatch):
    settings = Settings(
        app_name='DevIA Market Analysis Agent',
        environment='test',
        orchestration_mode='langgraph',
        report_synthesis_mode='openai_compatible',
        sentiment_analysis_mode='heuristic',
        llm_api_key='sk-test',
        llm_base_url='https://api.openai.com/v1',
        llm_model='gpt-4o-mini',
        llm_enabled=True,
        langfuse_public_key='pk-lf-test',
        langfuse_secret_key='sk-lf-secret',
        langfuse_base_url='https://us.cloud.langfuse.com',
        langfuse_enabled=True,
        sentiment_prompt_name='sentiment-analyzer',
        sentiment_prompt_label='production',
        report_prompt_name='market-analysis-report-generator',
        report_prompt_label='production',
        request_timeout_seconds=20.0,
    )

    class FakePrompt:
        name = 'market-analysis-report-generator'
        version = 3
        labels = ['production']

        def compile(self, **kwargs):
            assert 'request_json' in kwargs
            assert 'product_data_json' in kwargs
            assert 'sentiment_json' in kwargs
            assert 'trend_json' in kwargs
            return [{'role': 'system', 'content': 'summarize'}]

    class FakeLangfuseClient:
        def get_prompt(self, *args, **kwargs):
            assert kwargs['label'] == 'production'
            assert kwargs['type'] == 'chat'
            return FakePrompt()

    class FakeResponse:
        def __init__(self):
            self.choices = [type('Choice', (), {'message': type('Message', (), {'content': '{"executive_summary":"ok","key_findings":["a","b","c"]}'})()})]

    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs['langfuse_prompt'].name == 'market-analysis-report-generator'
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAIClient:
        chat = FakeChat()

    service = ReportNarrativeService(
        settings,
        langfuse_client=FakeLangfuseClient(),
        openai_client=FakeOpenAIClient(),
    )
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
            label='mixed',
            score=0.61,
            review_count=2,
            positive_signals=['reliable'],
            negative_signals=['expensive'],
            key_themes=['price perception'],
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

    assert draft.synthesis_mode == 'openai_compatible'
    assert draft.details['prompt_version'] == 3
