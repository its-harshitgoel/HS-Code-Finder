import asyncio

from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from models.schemas import ClassifyRequest, ClassifyResponse, HealthResponse
from utils.logger import get_logger

logger = get_logger("api")

router = APIRouter(prefix="/api")
limiter = Limiter(key_func=get_remote_address)

_classifier = None
_knowledge_base = None
_vector_search = None


def init_router(classifier, knowledge_base, vector_search):
    global _classifier, _knowledge_base, _vector_search
    _classifier = classifier
    _knowledge_base = knowledge_base
    _vector_search = vector_search


@router.post("/classify", response_model=ClassifyResponse)
@limiter.limit("20/minute")
async def classify(request: Request, body: ClassifyRequest) -> ClassifyResponse:
    if _classifier is None:
        raise HTTPException(status_code=503, detail="Classification service not ready.")

    logger.info("Classify request: session=%s", body.session_id or "new")

    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, _classifier.classify, body.session_id, body.message
        )
    except Exception as e:
        logger.error("Classification error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred. Please try again.")


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        dataset_loaded=_knowledge_base.is_loaded if _knowledge_base else False,
        index_built=_vector_search.is_built if _vector_search else False,
        entry_count=_knowledge_base.entry_count if _knowledge_base else 0,
    )
