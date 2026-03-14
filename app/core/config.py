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
    sentiment_analysis_mode: str
    llm_api_key: Optional[str]
    llm_base_url: str
    llm_model: str
    llm_enabled: bool
    langfuse_public_key: Optional[str]
    langfuse_secret_key: Optional[str]
    langfuse_base_url: str
    langfuse_enabled: bool
    sentiment_prompt_name: str
    sentiment_prompt_label: str
    report_prompt_name: str
    report_prompt_label: str
    request_timeout_seconds: float


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    langfuse_public_key = os.getenv('LANGFUSE_PUBLIC_KEY')
    langfuse_secret_key = os.getenv('LANGFUSE_SECRET_KEY')
    return Settings(
        app_name='DevIA Market Analysis Agent',
        environment=os.getenv('APP_ENV', 'local'),
        orchestration_mode=os.getenv('ORCHESTRATION_MODE', 'langgraph').strip().lower(),
        report_synthesis_mode=os.getenv('REPORT_SYNTHESIS_MODE', 'template').strip().lower(),
        sentiment_analysis_mode=os.getenv('SENTIMENT_ANALYSIS_MODE', 'heuristic').strip().lower(),
        llm_api_key=os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY'),
        llm_base_url=os.getenv('LLM_BASE_URL', 'https://api.openai.com/v1').rstrip('/'),
        llm_model=os.getenv('LLM_MODEL', 'gpt-4o-mini'),
        llm_enabled=_read_bool('LLM_ENABLED', False),
        langfuse_public_key=langfuse_public_key,
        langfuse_secret_key=langfuse_secret_key,
        langfuse_base_url=os.getenv('LANGFUSE_BASE_URL', 'https://cloud.langfuse.com').rstrip('/'),
        langfuse_enabled=bool(langfuse_public_key and langfuse_secret_key),
        sentiment_prompt_name=os.getenv('SENTIMENT_PROMPT_NAME', 'sentiment-analyzer').strip() or 'sentiment-analyzer',
        sentiment_prompt_label=os.getenv('SENTIMENT_PROMPT_LABEL', 'production').strip() or 'production',
        report_prompt_name=(
            os.getenv('REPORT_PROMPT_NAME', 'market-analysis-report-generator').strip()
            or 'market-analysis-report-generator'
        ),
        report_prompt_label=os.getenv('REPORT_PROMPT_LABEL', 'production').strip() or 'production',
        request_timeout_seconds=float(os.getenv('REQUEST_TIMEOUT_SECONDS', '20')),
    )
