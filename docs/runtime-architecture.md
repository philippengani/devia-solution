# Runtime Architecture

This diagram shows the full interaction flow of the solution when an analysis request runs, including FastAPI, orchestration, local tools, Langfuse, and the OpenAI-compatible provider path.

```mermaid
flowchart TD
    client["Client / Evaluator"]
    deploy["Docker Compose or local Uvicorn"]
    api["FastAPI app\n/app/main.py + /app/api/routes.py"]
    request["POST /analyze"]
    validation["Pydantic request validation\nAnalyzeRequest"]
    factory["Orchestrator factory\nlanggraph or native"]

    subgraph orchestration["Runtime orchestration"]
        request_span["Langfuse request trace\nstart_request()"]
        plan["plan_analysis"]
        product["ProductDataTool\nmocked catalog + seller profiles"]
        decision{"Customer reviews present?"}
        sentiment["SentimentAnalyzerTool"]
        heuristic["HeuristicSentimentAnalyzer\nlocal deterministic path"]
        sentiment_prompt["Langfuse prompt registry\nsentiment-analyzer"]
        sentiment_llm["Langfuse OpenAI client\nchat.completions"]
        trend["MarketTrendAnalyzerTool\npricing and demand analysis"]
        report["ReportGeneratorTool"]
        narrative["ReportNarrativeService"]
        template["Template narrative path"]
        report_prompt["Langfuse prompt registry\nmarket-analysis-report-generator"]
        report_llm["Langfuse OpenAI client\nchat.completions"]
        response["AnalyzeResponse JSON\nreport markdown + metadata + tool runs"]
        flush["Langfuse flush()"]
        memory["LangGraph MemorySaver\ncheckpointer"]
    end

    subgraph externals["External services"]
        openai["OpenAI-compatible API\nLLM_BASE_URL / model"]
        langfuse["Langfuse Cloud / self-hosted\nprompts + traces + generations"]
    end

    client --> deploy --> api --> request --> validation --> factory
    factory --> request_span
    factory --> memory
    request_span --> plan --> product --> decision
    product --> response
    decision -->|No| trend
    decision -->|Yes| sentiment
    sentiment -->|heuristic mode or fallback| heuristic --> trend
    sentiment -->|llm mode| sentiment_prompt --> langfuse
    sentiment_prompt --> sentiment_llm --> openai
    sentiment_llm --> langfuse
    sentiment_llm --> trend
    trend --> report --> narrative
    narrative -->|template mode or fallback| template --> response
    narrative -->|openai_compatible mode| report_prompt --> langfuse
    report_prompt --> report_llm --> openai
    report_llm --> langfuse
    report_llm --> response
    response --> flush --> langfuse
```

## What the graph means

- Every run starts at `POST /analyze`, where the request is validated and routed to either the LangGraph or native orchestrator.
- Product collection and trend analysis are always local and deterministic.
- Sentiment analysis is conditional: it is skipped when no reviews are provided.
- Sentiment can run in heuristic mode or in LLM mode, where the prompt is fetched from Langfuse and the generation is sent through the Langfuse OpenAI client.
- Report generation can stay on the local template path or use the OpenAI-compatible provider through a Langfuse-managed prompt.
- Langfuse is used for both observability and prompt management; failures in the LLM path fall back to the local deterministic path instead of failing the entire request.
