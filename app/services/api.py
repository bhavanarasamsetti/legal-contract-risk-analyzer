"""FastAPI application factory.

This module creates and configures the FastAPI application.  It is the
single entry point for the ASGI server:

.. code-block:: bash

    uvicorn app.services.api:app --reload --port 8000

Responsibilities:

- Define the application lifespan (startup and shutdown hooks).
- Construct and store the :class:`~app.analyzer.RiskAnalyzer` on
  ``app.state`` during startup.
- Mount all route routers.

This module does **not** define route handlers, request schemas, or
response schemas — those live in ``app/services/routes/``.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from app.services.routes import upload as upload_routes
from app.analyzer import RiskAnalyzer
from app.services.routes import analyze as analyze_routes
from app.services.routes import health as health_routes
from fastapi.middleware.cors import CORSMiddleware
# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown.

    **Startup** (before ``yield``):

    - Constructs a :class:`~app.analyzer.RiskAnalyzer` instance, which
      eagerly validates credentials and confirms the Pinecone index exists.
    - Stores the instance on ``application.state.analyzer`` so that route
      handlers can access it via ``request.app.state.analyzer``.

    If startup fails (missing API key, absent Pinecone index, network
    error), the exception propagates before ``yield`` and the server does
    not begin accepting requests.

    **Shutdown** (after ``yield``, on SIGTERM or KeyboardInterrupt):

    - Clears the analyzer reference so the garbage collector can reclaim
      resources promptly.

    Args:
        application: The :class:`~fastapi.FastAPI` instance being started.

    Yields:
        Nothing.  Control returns to FastAPI between the startup and
        shutdown phases.
    """
    application.state.analyzer = RiskAnalyzer(top_k=5)
  
    try:
        yield
    finally:
        application.state.analyzer = None


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Legal Contract Risk Analyzer",
    summary="Semantic question-answering API for ingested legal contracts.",
    description=(
        "Query the ingested contract corpus with natural-language questions. "
        "Relevant contract excerpts are retrieved from Pinecone via "
        "semantic similarity search, then passed to an OpenAI language model "
        "to generate a grounded, citation-backed answer.\n\n"
        "**Before using this API**, ingest contracts with:\n\n"
        "```bash\n"
        "python scripts/ingest.py --create-index\n"
        "```"
    ),
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------------------------------------------------------------
# Mount routers
# ---------------------------------------------------------------------------

app.include_router(health_routes.router)
app.include_router(analyze_routes.router)
app.include_router(upload_routes.router)
