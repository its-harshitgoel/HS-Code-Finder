import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware

from api.routes import init_router, limiter, router
from services.classifier import ClassificationEngine
from services.embedding import EmbeddingService
from services.hs_knowledge import HSKnowledgeBase
from services.llm_service import OpenAIService
from services.vector_search import VectorSearchService
from utils.logger import get_logger

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = get_logger("main")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
FRONTEND_DIR = BASE_DIR / "frontend"
CSV_PATH = DATA_DIR / "hs_codes.csv"

# Bump when embedding logic or indexed text changes — triggers automatic cache rebuild on startup.
# v6: removed stop-word filtering and parenthesis stripping from prepare_for_embedding
CACHE_VERSION = "v6-clean-embedding"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip().strip('"').strip("'")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set. LLM classification will not work.")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SERVE_FRONTEND = os.environ.get("SERVE_FRONTEND", "true").lower() == "true"

knowledge_base = HSKnowledgeBase()
embedding_service = EmbeddingService()
vector_search = VectorSearchService()
openai_service: OpenAIService | None = None
classifier: ClassificationEngine | None = None


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier, openai_service

    logger.info("HSCodeFinder — Starting up")

    knowledge_base.load(CSV_PATH)
    embedding_service.load_model()

    cache_version_file = CACHE_DIR / "cache_version.txt"
    cached_version = cache_version_file.read_text().strip() if cache_version_file.exists() else ""
    cache_valid = cached_version == CACHE_VERSION and vector_search.load_from_disk(CACHE_DIR)

    if not cache_valid:
        reason = "version mismatch" if cached_version else "no cache found"
        logger.info("Building fresh FAISS index (%s, target=%s)...", reason, CACHE_VERSION)
        entries_to_index = knowledge_base.get_subheadings()

        hierarchy: dict[str, list[str]] = {}
        for entry in entries_to_index:
            path = knowledge_base.get_hierarchy_path(entry.hs_code)
            descs = list(dict.fromkeys(e.description for e in path))
            if len(descs) > 1:
                hierarchy[entry.hs_code] = descs

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        vector_search.build_index(entries_to_index, embedding_service, hierarchy=hierarchy)
        vector_search.save_to_disk(CACHE_DIR)
        cache_version_file.write_text(CACHE_VERSION)

    openai_service = OpenAIService(api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL)
    openai_service.initialize()

    classifier = ClassificationEngine(
        knowledge_base, embedding_service, vector_search, openai_service
    )
    init_router(classifier, knowledge_base, vector_search)

    logger.info("HSCodeFinder — Ready at http://localhost:8001")

    yield

    logger.info("HSCodeFinder — Shutting down")


app = FastAPI(
    title="HSCodeFinder",
    description="HS Code Classification Assistant",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if SERVE_FRONTEND and FRONTEND_DIR.exists():
    app.mount("", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

    @app.get("/", include_in_schema=False)
    async def serve_frontend_root():
        index_path = FRONTEND_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {"message": "HSCodeFinder API is running."}
else:
    @app.get("/", include_in_schema=False)
    async def api_root():
        return {"message": "HSCodeFinder API is running.", "status": "ok"}
