from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.core.errors import AnalysisError
from app.models.schemas import AnalyzeRequest, AnalyzeResponse
from app.services.factory import get_orchestrator
from app.services.langgraph_orchestrator import LangGraphMarketAnalysisOrchestrator

router = APIRouter()


@router.get('/health')
def health() -> dict[str, str]:
    settings = get_settings()
    return {
        'status': 'ok',
        'orchestration_mode': settings.orchestration_mode,
        'report_synthesis_mode': settings.report_synthesis_mode,
    }


@router.post('/analyze', response_model=AnalyzeResponse)
def analyze_market(request: AnalyzeRequest) -> AnalyzeResponse:
    try:
        orchestrator = get_orchestrator()
        return orchestrator.run(request)
    except AnalysisError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Unexpected error: {exc}') from exc


@router.get('/workflow/diagram')
def workflow_diagram() -> dict[str, str]:
    orchestrator = LangGraphMarketAnalysisOrchestrator()
    return {
        'orchestration_mode': orchestrator.orchestration_mode,
        'mermaid': orchestrator.render_mermaid(),
        'ascii': orchestrator.render_ascii(),
    }
