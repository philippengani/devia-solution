import pytest

from app.models.schemas import AnalysisPlan, AnalyzeRequest
from app.tools.product_data_tool import ProductDataTool
from app.tools.report_tool import ReportGeneratorTool
from app.tools.sentiment_tool import SentimentAnalyzerTool
from app.tools.trend_tool import MarketTrendAnalyzerTool
from app.services.report_narrative import ReportNarrativeService
from app.core.config import get_settings


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
    tool = SentimentAnalyzerTool()
    result = tool.run(['Great and reliable', 'Too expensive'])
    assert result.label == 'mixed'
    assert 'great' in result.positive_signals
    assert 'expensive' in result.negative_signals
    assert 'price perception' in result.key_themes


def test_sentiment_tool_returns_insufficient_data_without_reviews():
    tool = SentimentAnalyzerTool()
    result = tool.run(None)
    assert result.label == 'insufficient_data'
    assert result.score is None
    assert result.review_count == 0


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
