"""
API Routes for the HSCodeFinder application.

Purpose: Defines FastAPI endpoints that connect the frontend to backend services.
Endpoints:
    POST /api/classify — Accept product description, return candidates/questions/result.
    GET  /api/health   — Health check with system status.
"""

from fastapi import APIRouter, HTTPException, Request

from backend.models.schemas import ClassifyRequest, ClassifyResponse, HealthResponse
from backend.utils.logger import get_logger
from backend.utils.rate_limit import is_rate_limited

logger = get_logger("api")

router = APIRouter(prefix="/api")

# These will be injected by main.py on startup
_classifier = None
_knowledge_base = None
_vector_search = None


def init_router(classifier, knowledge_base, vector_search):
    """Inject service dependencies into the router.

    Called during app startup after all services are initialized.
    """
    global _classifier, _knowledge_base, _vector_search
    _classifier = classifier
    _knowledge_base = knowledge_base
    _vector_search = vector_search


@router.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest, http_request: Request) -> ClassifyResponse:
    """Classify a product description or process a follow-up answer.

    New conversations: Send session_id=null with a product description.
    Follow-ups: Send the existing session_id with the answer.

    Returns candidates, a clarifying question, or the final HS code result.
    """
    if _classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Classification service is not ready. Please wait for initialization.",
        )

    client_ip = (http_request.client.host if http_request.client else None) or "unknown"
    if is_rate_limited(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait a minute and try again.",
        )

    logger.info(
        "Classify request: session=%s, message_length=%d",
        request.session_id or "new",
        len(request.message),
    )

    try:
        response = _classifier.classify(
            session_id=request.session_id,
            message=request.message,
        )
        return response
    except Exception as e:
        logger.error("Classification error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred during classification. Please try again.",
        )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint with system status."""
    return HealthResponse(
        status="ok",
        dataset_loaded=_knowledge_base.is_loaded if _knowledge_base else False,
        index_built=_vector_search.is_built if _vector_search else False,
        entry_count=_knowledge_base.entry_count if _knowledge_base else 0,
    )
