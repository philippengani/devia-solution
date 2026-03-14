import json
import pytest

from app.core.config import get_settings
from app.models.schemas import AnalysisPlan, AnalyzeRequest
from app.tools.sentiment_tool import SentimentTraceContext
from app.tools.product_data_tool import ProductDataTool
from app.tools.report_tool import ReportGeneratorTool
from app.tools.sentiment_tool import SentimentAnalyzerTool
from app.tools.trend_tool import MarketTrendAnalyzerTool
from app.services.report_narrative import ReportNarrativeService


def test_product_data_tool_uses_selected_competitors():
    tool = ProductDataTool()
    request = AnalyzeRequest(product_name='iPhone 15', competitors=['Amazon', 'Best Buy', 'Walmart'])
    plan = AnalysisPlan(
        objective='Test plan',
        selected_competitors=['Amazon', 'Best Buy', 'Walmart'],
        requires_sentiment=True,
        seller_selection_reason='Provided by request',
        steps=['collect_product_data'],
    )
    result = tool.run(request, plan)
    assert len(result) == 3
    assert [item.source for item in result] == ['Amazon', 'Best Buy', 'Walmart']
    assert all(item.search_url.startswith('https://mock.market.local/') for item in result)


def test_sentiment_tool_detects_mixed_feedback_and_themes():
    get_settings.cache_clear()
    tool = SentimentAnalyzerTool()
    result = tool.run(['Great and reliable', 'Too expensive'])
    assert result.output.label == 'mixed'
    assert 'great' in result.output.positive_signals
    assert 'expensive' in result.output.negative_signals
    assert 'price perception' in result.output.key_themes
    assert result.details['effective_mode'] == 'heuristic'


def test_sentiment_tool_returns_insufficient_data_without_reviews():
    tool = SentimentAnalyzerTool()
    result = tool.run(None)
    assert result.output.label == 'insufficient_data'
    assert result.output.score is None
    assert result.output.review_count == 0


def test_sentiment_tool_uses_llm_when_configured(monkeypatch):
    monkeypatch.setenv('LLM_ENABLED', 'true')
    monkeypatch.setenv('LLM_API_KEY', 'sk-test')
    monkeypatch.setenv('LLM_MODEL', 'gpt-4o-mini')
    monkeypatch.setenv('SENTIMENT_ANALYSIS_MODE', 'llm')
    monkeypatch.setenv('LANGFUSE_PUBLIC_KEY', 'pk-lf-test')
    monkeypatch.setenv('LANGFUSE_SECRET_KEY', 'sk-lf-secret')
    monkeypatch.setenv('LANGFUSE_BASE_URL', 'https://us.cloud.langfuse.com')
    get_settings.cache_clear()

    class FakePrompt:
        name = 'sentiment-analyzer'
        version = 7
        labels = ['production']

        def compile(self, **kwargs):
            assert kwargs['review_count'] == 2
            assert json.loads(kwargs['reviews_json']) == ['Great product', 'Too expensive']
            return [{'role': 'system', 'content': 'analyze'}]

    class FakeLangfuseClient:
        def get_prompt(self, *args, **kwargs):
            assert kwargs['label'] == 'production'
            assert kwargs['type'] == 'chat'
            return FakePrompt()

    class FakeResponse:
        def __init__(self):
            self.choices = [type('Choice', (), {'message': type('Message', (), {'content': json.dumps({
                'label': 'mixed',
                'score': 0.61,
                'positive_signals': ['reliable'],
                'negative_signals': ['expensive'],
                'key_themes': ['price perception'],
            })})()})]

    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs['langfuse_prompt'].name == 'sentiment-analyzer'
            assert kwargs['trace_id'] == 'trace-123'
            assert kwargs['parent_observation_id'] == 'span-456'
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAIClient:
        chat = FakeChat()

    tool = SentimentAnalyzerTool(
        get_settings(),
        langfuse_client=FakeLangfuseClient(),
        openai_client=FakeOpenAIClient(),
    )
    result = tool.run(
        ['Great product', 'Too expensive'],
        trace_context=SentimentTraceContext(trace_id='trace-123', parent_observation_id='span-456'),
    )

    assert result.output.label == 'mixed'
    assert result.output.review_count == 2
    assert result.details['effective_mode'] == 'llm'
    assert result.details['prompt_version'] == 7


def test_sentiment_tool_falls_back_to_heuristic_when_llm_fails(monkeypatch):
    monkeypatch.setenv('LLM_ENABLED', 'true')
    monkeypatch.setenv('LLM_API_KEY', 'sk-test')
    monkeypatch.setenv('SENTIMENT_ANALYSIS_MODE', 'llm')
    monkeypatch.setenv('LANGFUSE_PUBLIC_KEY', 'pk-lf-test')
    monkeypatch.setenv('LANGFUSE_SECRET_KEY', 'sk-lf-secret')
    get_settings.cache_clear()

    class BrokenLangfuseClient:
        def get_prompt(self, *args, **kwargs):
            raise RuntimeError('prompt unavailable')

    tool = SentimentAnalyzerTool(get_settings(), langfuse_client=BrokenLangfuseClient(), openai_client=object())
    result = tool.run(['Great and reliable', 'Too expensive'])

    assert result.output.label == 'mixed'
    assert result.details['fallback_mode'] == 'heuristic'
    assert result.warnings


def test_trend_tool_computes_spread_and_price_band():
    request = AnalyzeRequest(product_name='Nike Air Max')
    plan = AnalysisPlan(
        objective='Test plan',
        selected_competitors=ProductDataTool.default_competitors('Nike Air Max'),
        requires_sentiment=False,
        seller_selection_reason='Default sellers',
        steps=['collect_product_data'],
    )
    product_data = ProductDataTool().run(request, plan)
    trend = MarketTrendAnalyzerTool().run(product_data)
    assert trend.max_price >= trend.min_price
    assert trend.price_spread == round(trend.max_price - trend.min_price, 2)
    assert trend.recommended_price_floor <= trend.recommended_price_ceiling


def test_trend_tool_requires_product_data():
    with pytest.raises(ValueError):
        MarketTrendAnalyzerTool().run([])


def test_report_tool_omits_recommendations_when_requested():
    request = AnalyzeRequest(product_name='PlayStation 5', include_recommendations=False)
    plan = AnalysisPlan(
        objective='Test plan',
        selected_competitors=ProductDataTool.default_competitors('PlayStation 5'),
        requires_sentiment=False,
        seller_selection_reason='Default sellers',
        steps=['collect_product_data', 'skip_sentiment', 'analyze_trend', 'generate_report'],
    )
    product_data = ProductDataTool().run(request, plan)
    sentiment = SentimentAnalyzerTool().empty_output('No customer reviews supplied.')
    trend = MarketTrendAnalyzerTool().run(product_data)
    tool = ReportGeneratorTool(ReportNarrativeService(get_settings()))

    report, warnings = tool.run(request, plan, product_data, sentiment, trend)

    assert report.recommendations == []
    assert 'Recommendations intentionally omitted' in report.markdown
    assert warnings == []
