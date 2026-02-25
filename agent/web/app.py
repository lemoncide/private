from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.core.agent import LangGraphAgent


class InvokeRequest(BaseModel):
    query: str


class InvokeResponse(BaseModel):
    response: str | None
    status: str | None
    error: str | None
    steps: List[Dict[str, Any]]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = LangGraphAgent()


@app.post("/api/invoke", response_model=InvokeResponse)
async def invoke(req: InvokeRequest) -> InvokeResponse:
    result = await agent.ainvoke(req.query)
    trace = result.get("trace") or {}
    past_steps = trace.get("past_steps") or []
    repair_history = trace.get("repair_history") or []

    repair_index: Dict[str, List[Dict[str, Any]]] = {}
    for r in repair_history:
        step_id = r.get("step_id")
        if not step_id:
            continue
        repair_index.setdefault(step_id, []).append(r)

    steps_view: List[Dict[str, Any]] = []
    for r in past_steps:
        if hasattr(r, "model_dump"):
            step_dict = r.model_dump()
        else:
            step_dict = dict(r)
        meta = step_dict.get("meta") or {}
        step_id = step_dict.get("step_id")
        steps_view.append(
            {
                "step_id": step_id,
                "status": step_dict.get("status"),
                "tool": meta.get("tool"),
                "error_type": step_dict.get("error_type"),
                "error": step_dict.get("error"),
                "repaired": bool(repair_index.get(step_id or "")),
            }
        )

    return InvokeResponse(
        response=result.get("response"),
        status=result.get("status"),
        error=trace.get("error"),
        steps=steps_view,
    )


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("agent/web/static/index.html")


app.mount("/static", StaticFiles(directory="agent/web/static"), name="static")

