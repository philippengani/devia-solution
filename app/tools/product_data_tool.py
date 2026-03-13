from __future__ import annotations

from urllib.parse import quote_plus

from app.models.schemas import AnalysisPlan, AnalyzeRequest, ProductObservation


class ProductDataTool:
    """Deterministic mocked market data collector used by the orchestrator."""

    _CATALOG = {
        'iphone 15': {
            'title': 'Apple iPhone 15 128GB',
            'base_price': 1129.0,
            'default_competitors': ['Amazon', 'Best Buy', 'Walmart'],
        },
        'nike air max': {
            'title': 'Nike Air Max 270',
            'base_price': 179.0,
            'default_competitors': ['Nike', 'Foot Locker', 'SportChek'],
        },
        'playstation 5': {
            'title': 'Sony PlayStation 5 Slim',
            'base_price': 649.0,
            'default_competitors': ['Amazon', 'GameStop', 'Walmart'],
        },
    }

    _SELLER_PROFILES = {
        'amazon': {'offset': -18.0, 'rating': 4.6, 'reviews': 1360, 'availability': 'in_stock'},
        'best buy': {'offset': 0.0, 'rating': 4.4, 'reviews': 810, 'availability': 'limited_stock'},
        'walmart': {'offset': 14.0, 'rating': 4.1, 'reviews': 540, 'availability': 'in_stock'},
        'nike': {'offset': 6.0, 'rating': 4.7, 'reviews': 920, 'availability': 'in_stock'},
        'foot locker': {'offset': -7.0, 'rating': 4.3, 'reviews': 470, 'availability': 'limited_stock'},
        'sportchek': {'offset': 4.0, 'rating': 4.2, 'reviews': 250, 'availability': 'in_stock'},
        'gamestop': {'offset': -11.0, 'rating': 4.4, 'reviews': 390, 'availability': 'preorder'},
    }

    _MARKET_CONFIG = {
        'CA': {'multiplier': 1.0, 'currency': 'CAD'},
        'US': {'multiplier': 0.74, 'currency': 'USD'},
        'UK': {'multiplier': 0.58, 'currency': 'GBP'},
        'EU': {'multiplier': 0.68, 'currency': 'EUR'},
    }

    @classmethod
    def default_competitors(cls, product_name: str) -> list[str]:
        product = cls._CATALOG.get(product_name.strip().lower())
        if product:
            return list(product['default_competitors'])
        return ['Amazon', 'Best Buy', 'Walmart']

    @classmethod
    def supported_markets(cls) -> set[str]:
        return set(cls._MARKET_CONFIG)

    def run(self, request: AnalyzeRequest, plan: AnalysisPlan) -> list[ProductObservation]:
        normalized_name = request.product_name.strip().lower()
        if not normalized_name:
            raise ValueError('product_name must not be empty')

        product = self._CATALOG.get(
            normalized_name,
            {
                'title': request.product_name,
                'base_price': 249.0,
                'default_competitors': ['Amazon', 'Best Buy', 'Walmart'],
            },
        )

        sellers = plan.selected_competitors or product['default_competitors']
        market_config = self._MARKET_CONFIG.get(request.market, {'multiplier': 0.74, 'currency': 'USD'})
        base_price = product['base_price'] * market_config['multiplier']

        observations: list[ProductObservation] = []
        for index, seller in enumerate(sellers[:3]):
            seller_key = seller.casefold()
            seller_profile = self._SELLER_PROFILES.get(
                seller_key,
                {
                    'offset': float(index * 6),
                    'rating': 4.2,
                    'reviews': 320 + index * 90,
                    'availability': 'in_stock',
                },
            )
            observations.append(
                ProductObservation(
                    source=seller,
                    product_title=product['title'],
                    price=round(base_price + seller_profile['offset'], 2),
                    currency=market_config['currency'],
                    availability=seller_profile['availability'],
                    rating=round(float(seller_profile['rating']), 1),
                    review_count=int(seller_profile['reviews']),
                    search_url=f'https://mock.market.local/{quote_plus(seller.lower())}/{quote_plus(request.product_name.lower())}',
                )
            )

        return observations
