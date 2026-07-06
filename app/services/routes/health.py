"""Health-check route.

``GET /health`` is a shallow liveness and readiness probe.  It reports
whether the application started successfully and the
:class:`~app.analyzer.RiskAnalyzer` is ready to serve requests.

The check is intentionally **shallow**: it reads ``app.state.analyzer``
without making any calls to OpenAI or Pinecone.  This keeps the probe
fast (< 1 ms) and free of external-service costs, while still correctly
signalling an unready state during the startup window.

Kubernetes usage::

    livenessProbe:
      httpGet:
        path: /health
        port: 8000
    readinessProbe:
      httpGet:
        path: /health
        port: 8000
"""

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Response body for ``GET /health``.

    Attributes:
        status: Human-readable service status.  ``"ok"`` when the analyzer
            is ready; ``"starting"`` during the startup window.
        analyzer_ready: ``True`` when :class:`~app.analyzer.RiskAnalyzer`
            has been successfully initialised and is ready to serve
            requests.
    """

    status: str
    analyzer_ready: bool


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and readiness probe",
    description=(
        "Returns ``200 OK`` when the service is fully initialised and ready "
        "to handle analysis requests.  Returns ``503 Service Unavailable`` "
        "during the startup window before the analyzer is ready."
    ),
)
async def health(request: Request) -> JSONResponse:
    """Return the current service health status.

    Checks whether ``app.state.analyzer`` has been set by the application
    lifespan.  No external API calls are made.

    Args:
        request: The incoming HTTP request, used to access ``app.state``.

    Returns:
        A :class:`~fastapi.responses.JSONResponse` with HTTP ``200`` when
        the analyzer is ready, or ``503`` when startup is still in progress.
    """
    analyzer_ready = getattr(request.app.state, "analyzer", None) is not None

    if analyzer_ready:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "ok", "analyzer_ready": True},
        )

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": "starting", "analyzer_ready": False},
    )
