"""
TravelBot FastAPI backend.

Run with:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Docs available at:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tourist_agent.graph import init_graph, cleanup_graph
from tourist_agent.tracing import init_tracing
from api.routes.chat import router as chat_router
from api.routes.sessions import router as sessions_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise async resources on startup."""
    init_tracing()   # LangFuse SDK-level auto-instrumentation (async-safe)
    await init_graph()
    yield
    await cleanup_graph()   # release Postgres connection pool


app = FastAPI(
    title="TravelBot API",
    description="AI travel agent for Ladakh & Himachal Pradesh tour packages.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(sessions_router)
app.include_router(chat_router)


@app.get("/health", tags=["health"])
def health():
    """Liveness check."""
    return {"status": "ok"}
