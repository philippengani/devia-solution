from __future__ import annotations

from statistics import mean

from app.models.schemas import ProductObservation, TrendOutput


class MarketTrendAnalyzerTool:
    def run(self, product_data: list[ProductObservation]) -> TrendOutput:
        if not product_data:
            raise ValueError('product_data is required for trend analysis')

        prices = [item.price for item in product_data]
        ratings = [item.rating for item in product_data]
        reviews = [item.review_count for item in product_data]

        avg_price = round(mean(prices), 2)
        min_price = round(min(prices), 2)
        max_price = round(max(prices), 2)
        price_spread = round(max_price - min_price, 2)
        avg_rating = round(mean(ratings), 2)
        total_reviews = sum(reviews)

        if price_spread <= max(avg_price * 0.08, 10):
            direction = 'stable'
        else:
            direction = 'volatile'

        if price_spread <= max(avg_price * 0.05, 12):
            competitiveness = 'high'
        elif price_spread <= max(avg_price * 0.1, 20):
            competitiveness = 'medium'
        else:
            competitiveness = 'low'

        if avg_rating >= 4.4 and total_reviews >= 1800:
            demand_signal = 'high'
        elif avg_rating >= 4.1 and total_reviews >= 900:
            demand_signal = 'medium'
        else:
            demand_signal = 'low'

        recommended_price_floor = round(max(min_price, avg_price - max(price_spread * 0.25, avg_price * 0.01)), 2)
        recommended_price_ceiling = round(
            min(max_price, avg_price + max(price_spread * 0.15, avg_price * 0.01)),
            2,
        )

        insight = (
            f'Observed pricing is {direction} with {competitiveness} competitive pressure. '
            f'Demand looks {demand_signal} based on average rating {avg_rating:.2f} across {total_reviews} reviews.'
        )

        return TrendOutput(
            direction=direction,
            avg_price=avg_price,
            min_price=min_price,
            max_price=max_price,
            price_spread=price_spread,
            competitiveness=competitiveness,
            demand_signal=demand_signal,
            recommended_price_floor=recommended_price_floor,
            recommended_price_ceiling=recommended_price_ceiling,
            insight=insight,
        )
