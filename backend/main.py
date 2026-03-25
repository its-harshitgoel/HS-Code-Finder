"""
HSCodeFinder — FastAPI Application Entry Point.

Purpose: Initializes all services on startup (dataset, embeddings, FAISS index),
         mounts the API router, and serves the frontend static files.

Startup sequence:
    1. Load HS dataset from CSV
    2. Load embedding model
    3. Build FAISS vector index
    4. Initialize classification engine
    5. Serve frontend and API
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.api.routes import init_router, router
from backend.services.classifier import ClassificationEngine
from backend.services.embedding import EmbeddingService
from backend.services.hs_knowledge import HSKnowledgeBase
from backend.services.llm_service import GeminiService
from backend.services.vector_search import VectorSearchService
from backend.utils.logger import get_logger

# Load .env file from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = get_logger("main")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FRONTEND_DIR = BASE_DIR / "frontend"
CSV_PATH = DATA_DIR / "hs_codes.csv"

# Global service instances
knowledge_base = HSKnowledgeBase()
embedding_service = EmbeddingService()
vector_search = VectorSearchService()
gemini_service: GeminiService | None = None
classifier: ClassificationEngine | None = None

def _load_gemini_api_key() -> str:
    """Load and sanitize GEMINI_API_KEY from environment."""
    raw = os.getenv("GEMINI_API_KEY")
    if raw is None:
        return ""

    value = raw.strip().strip('"').strip("'")

    # Prevent accidental placeholder values from being treated as real secrets.
    placeholders = {
        "your_gemini_api_key_here",
        "your_key_here",
        "changeme",
        "replace_me",
    }
    if not value or value.lower() in placeholders:
        return ""

    return value


# Gemini API key — loaded from .env, never hardcoded
GEMINI_API_KEY = _load_gemini_api_key()
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY missing or placeholder. LLM reasoning will be limited until it is set.")


def _parse_csv_env(name: str, default: str) -> list[str]:
    raw = os.environ.get(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


ALLOWED_ORIGINS = _parse_csv_env("ALLOWED_ORIGINS", "http://localhost:8001,http://127.0.0.1:8001")
ALLOWED_HOSTS = _parse_csv_env("ALLOWED_HOSTS", "localhost,127.0.0.1")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all services on startup."""
    global classifier, gemini_service

    logger.info("=" * 60)
    logger.info("HSCodeFinder — Starting up")
    logger.info("=" * 60)

    # Step 1: Load HS dataset
    logger.info("Step 1/5: Loading HS dataset...")
    knowledge_base.load(CSV_PATH)

    # Step 2: Load embedding model
    logger.info("Step 2/5: Loading embedding model...")
    embedding_service.load_model()

    # Step 3: Build FAISS index (index subheadings + headings for broader coverage)
    logger.info("Step 3/5: Building vector index...")
    entries_to_index = knowledge_base.get_subheadings() + knowledge_base.get_headings()
    vector_search.build_index(entries_to_index, embedding_service)

    # Step 4: Initialize Gemini LLM
    logger.info("Step 4/5: Initializing Gemini LLM...")
    gemini_service = GeminiService(api_key=GEMINI_API_KEY)
    gemini_service.initialize()

    # Step 5: Initialize classification engine
    logger.info("Step 5/5: Initializing classification engine...")
    classifier = ClassificationEngine(
        knowledge_base, embedding_service, vector_search, gemini_service
    )
    init_router(classifier, knowledge_base, vector_search)

    logger.info("=" * 60)
    logger.info("HSCodeFinder — Ready! Serving at http://localhost:8001")
    logger.info("=" * 60)

    yield

    logger.info("HSCodeFinder — Shutting down")


# Create FastAPI app
app = FastAPI(
    title="HSCodeFinder",
    description="Intelligent HS Code Classification Assistant",
    version="1.0.0",
    lifespan=lifespan,
)

# Trusted host middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request, call_next):
    """Add basic hardening headers to every response."""
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    return response

# Register API routes
app.include_router(router)

# Mount frontend static files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def serve_frontend():
    """Serve the frontend index.html."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "HSCodeFinder API is running. Frontend not found."}
