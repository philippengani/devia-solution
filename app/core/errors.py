from __future__ import annotations


class AnalysisError(Exception):
    status_code = 500


class ConfigurationError(AnalysisError):
    status_code = 500


class ToolExecutionError(AnalysisError):
    status_code = 500

    def __init__(self, tool_name: str, message: str) -> None:
        super().__init__(f'{tool_name} failed: {message}')
        self.tool_name = tool_name
