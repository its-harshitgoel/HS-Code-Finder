from __future__ import annotations
import re
import uuid as _uuid
from pydantic import BaseModel, field_validator


class HSEntry(BaseModel):
    section: str
    hs_code: str
    description: str
    parent: str
    level: int


class Candidate(BaseModel):
    hs_code: str
    description: str
    section: str
    level: int
    parent: str
    similarity_score: float


class FinalResult(BaseModel):
    hs_code: str
    description: str
    explanation: str


class SessionState(BaseModel):
    session_id: str
    original_query: str
    candidates: list[Candidate] = []
    questions_asked: int = 0
    llm_history: list[dict] = []
    awaiting_product_description: bool = False
    retrieval_retried: bool = False        # True after first REFINE retry (LLM-suggested terms)
    enriched_retrieval_done: bool = False  # True after second REFINE retry (query + user answers)
    confirmed_qa: list[dict] = []          # [{q: "question text", a: "user answer"}, ...]


class ClassifyRequest(BaseModel):
    session_id: str | None = None
    message: str

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str | None) -> str | None:
        if v is not None:
            try:
                _uuid.UUID(v)
            except ValueError:
                raise ValueError("session_id must be a valid UUID")
        return v

    @field_validator("message")
    @classmethod
    def sanitize_message(cls, v: str) -> str:
        cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", v)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            raise ValueError("message cannot be empty")
        return cleaned[:1000]


class ClassifyResponse(BaseModel):
    session_id: str
    type: str
    message: str
    candidates: list[Candidate]
    final_result: FinalResult | None = None
    options: list[str] | None = None


class HealthResponse(BaseModel):
    status: str
    dataset_loaded: bool
    index_built: bool
    entry_count: int
