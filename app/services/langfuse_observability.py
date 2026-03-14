from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Optional

from langfuse import Langfuse, propagate_attributes
from langfuse.types import TraceContext

from app.core.config import Settings


class LangfuseObservability:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = self._build_client()

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def get_client(self):
        return self._client

    def _build_client(self):
        if not self.settings.langfuse_enabled:
            return None
        return Langfuse(
            public_key=self.settings.langfuse_public_key,
            secret_key=self.settings.langfuse_secret_key,
            base_url=self.settings.langfuse_base_url,
            tracing_enabled=True,
            environment=self.settings.environment,
        )

    def start_request(self, *, analysis_id: str, request, orchestration_mode: str):
        client = self.get_client()
        if client is None:
            return nullcontext(None)

        metadata = {
            'analysis_id': analysis_id,
            'product_name': request.product_name,
            'market': request.market,
            'orchestration_mode': orchestration_mode,
        }
        tags = [orchestration_mode, request.market.lower()]

        propagation = propagate_attributes(
            session_id=analysis_id,
            metadata=metadata,
            tags=tags,
            trace_name='market-analysis',
        )
        propagation.__enter__()
        observation = client.start_as_current_observation(
            name='market-analysis',
            as_type='span',
            input={
                'product_name': request.product_name,
                'market': request.market,
                'competitor_count': len(request.competitors),
                'has_reviews': bool(request.customer_reviews),
            },
            metadata=metadata,
        )
        span = observation.__enter__()
        return _CombinedContextManager(propagation, observation, span)

    def start_step(self, *, step_name: str, input_data: Optional[Any] = None, metadata: Optional[dict[str, Any]] = None):
        client = self.get_client()
        if client is None:
            return nullcontext(None)
        return client.start_as_current_observation(
            name=step_name,
            as_type='tool',
            input=input_data,
            metadata=metadata,
        )

    def flush(self) -> None:
        client = self.get_client()
        if client is not None:
            client.flush()

    @staticmethod
    def trace_context_from_observation(observation: Any) -> Optional[TraceContext]:
        if observation is None:
            return None
        trace_id = getattr(observation, 'trace_id', None)
        parent_span_id = getattr(observation, 'id', None)
        if not trace_id or not parent_span_id:
            return None
        return TraceContext(trace_id=trace_id, parent_span_id=parent_span_id)


class _CombinedContextManager:
    def __init__(self, propagation, observation, span) -> None:
        self._propagation = propagation
        self._observation = observation
        self._span = span

    def __enter__(self):
        return self._span

    def __exit__(self, exc_type, exc, tb):
        suppress = self._observation.__exit__(exc_type, exc, tb)
        self._propagation.__exit__(exc_type, exc, tb)
        return suppress
