from __future__ import annotations

from app.core.config import get_settings
from app.services.langgraph_orchestrator import LangGraphMarketAnalysisOrchestrator
from app.services.native_orchestrator import NativeMarketAnalysisOrchestrator


def get_orchestrator():
    mode = get_settings().orchestration_mode
    if mode == 'native':
        return NativeMarketAnalysisOrchestrator()
    return LangGraphMarketAnalysisOrchestrator()
