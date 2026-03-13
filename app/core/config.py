from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from typing import Optional


def _read_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {'1', 'true', 'yes', 'on'}


@dataclass(frozen=True)
class Settings:
    app_name: str
    environment: str
    orchestration_mode: str
    report_synthesis_mode: str
    llm_api_key: Optional[str]
    llm_base_url: str
    llm_model: str
    llm_enabled: bool
    request_timeout_seconds: float


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        app_name='DevIA Market Analysis Agent',
        environment=os.getenv('APP_ENV', 'local'),
        orchestration_mode=os.getenv('ORCHESTRATION_MODE', 'langgraph').strip().lower(),
        report_synthesis_mode=os.getenv('REPORT_SYNTHESIS_MODE', 'template').strip().lower(),
        llm_api_key=os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY'),
        llm_base_url=os.getenv('LLM_BASE_URL', 'https://api.openai.com/v1').rstrip('/'),
        llm_model=os.getenv('LLM_MODEL', 'gpt-4o-mini'),
        llm_enabled=_read_bool('LLM_ENABLED', False),
        request_timeout_seconds=float(os.getenv('REQUEST_TIMEOUT_SECONDS', '20')),
    )
