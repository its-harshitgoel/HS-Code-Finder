# HS Code Finder — Product & Flow Documentation

## 1) Product Overview

HS Code Finder is an AI-assisted classification tool that helps users identify the most appropriate **6-digit Harmonized System (HS) code** for a product.

### Core value

- Accept plain-language product descriptions.
- Retrieve best HS candidates using semantic search.
- Use LLM reasoning to ask clarifying questions.
- Return final HS code with explanation.

---

## 2) End-to-End User Flow

1. User opens the web app at `/`.
2. User types a product description (example: “frozen shrimp seafood”).
3. Frontend sends request to `POST /api/classify`.
4. Backend validates and sanitizes input.
5. Classification engine either:
   - starts a new session (if `session_id` is null), or
   - continues existing session.
6. System creates embeddings for user text.
7. FAISS vector search finds top HS candidates.
8. Gemini receives candidate context + conversation context.
9. Gemini returns either:
   - a short clarifying question, or
   - a final classification in structured format.
10. Backend parses response and returns JSON to frontend.
11. Frontend renders either:

- follow-up question, or
- final result card with hierarchy path.

---

## 3) Backend Runtime Flow (Startup)

On application startup (`backend/main.py` lifespan):

1. Load `.env` variables.
2. Load HS dataset (`data/hs_codes.csv`) into memory.
3. Load embedding model (`all-MiniLM-L6-v2`).
4. Build FAISS index from headings + subheadings.
5. Initialize Gemini client.
6. Initialize classification engine and inject dependencies into API router.
7. Serve API + static frontend.

---

## 4) API Flow Details

### `POST /api/classify`

Request body:

- `session_id` (optional UUID)
- `message` (required string)

Flow:

1. Endpoint checks service readiness.
2. Rate limiter checks client request volume.
3. Request model sanitizes and validates input.
4. Classification engine processes query.
5. Returns `ClassifyResponse` with type:
   - `question` or
   - `result`

### `GET /api/health`

Returns status flags:

- dataset loaded
- vector index built
- total entry count

---

## 5) Classification Engine Logic

File: `backend/services/classifier.py`

### New session

- Create `session_id`.
- Save initial user message.
- Embed query and search top-k candidates.
- Send candidates to Gemini.
- Parse Gemini output:
  - if `RESULT:`, build final result
  - else return clarifying question

### Existing session

- Append user follow-up answer.
- Re-run semantic search using combined query (`original + answer`).
- Send updated context to Gemini.
- Parse and return question/result.

### Safeguards

- Empty message protection.
- Max question cap (`MAX_QUESTIONS`) to avoid infinite loops.
- Result fallback if search/LLM flow degrades.

---

## 6) AI and Search Components

### Embedding Service (`backend/services/embedding.py`)

- Uses sentence-transformers `all-MiniLM-L6-v2`.
- Normalized vectors for cosine-like similarity.
- Supports single and batch encoding.

### Vector Search (`backend/services/vector_search.py`)

- Uses FAISS `IndexFlatIP`.
- Searches nearest HS description embeddings.
- Returns ranked `Candidate` objects with similarity score.

### LLM Service (`backend/services/llm_service.py`)

- Uses Google Gemini via `google-genai` SDK.
- Has system prompt with strict classification behavior.
- Retry/backoff for transient API failures.
- Prompt injection hardening:
  - sanitize untrusted user text,
  - treat user text as data, not instructions,
  - bounded output handling.

---

## 7) Data Layer

### Source

- `data/hs_codes.csv`
- Includes section, code, description, parent, level.

### Knowledge Base (`backend/services/hs_knowledge.py`)

- Loads CSV and validates schema.
- Stores indexed structures:
  - by code
  - by parent
- Supports hierarchy traversal (chapter → heading → subheading).

---

## 8) Frontend Flow

Files:

- `frontend/index.html`
- `frontend/style.css`
- `frontend/app.js`

Behavior:

1. Capture user message.
2. Call `/api/classify` via fetch.
3. Render user bubble immediately.
4. Show typing indicator while waiting.
5. Render assistant question or result card.
6. Persist conversation via `session_id` in memory.
7. Reset session after final result.

UI result card includes:

- HS code
- Description
- Parsed hierarchy path

---

## 9) Security Model

### Secrets

- API key is loaded from environment (`GEMINI_API_KEY`).
- `.env` is ignored in VCS.
- `.env.example` is template-only.

### Request safety

- Pydantic schema validation.
- Message sanitization (control-char cleanup, whitespace normalization, length constraints).
- UUID validation for session id.

### API abuse protection

- In-memory per-IP rate limiting on classify endpoint.
- Configurable via env:
  - `RATE_LIMIT_WINDOW_SECONDS`
  - `RATE_LIMIT_MAX_REQUESTS`

### Web/API hardening

- CORS allowlist (`ALLOWED_ORIGINS`).
- Trusted host allowlist (`ALLOWED_HOSTS`).
- Security headers:
  - `X-Content-Type-Options: nosniff`
  - `X-Frame-Options: DENY`
  - `Referrer-Policy: no-referrer`
  - `Permissions-Policy` restrictions

### Logging safety

- Avoid logging raw user product descriptions.
- Log metadata (length/session) instead of content.

---

## 10) Environment Variables

Required:

- `GEMINI_API_KEY`

Recommended security/runtime:

- `ALLOWED_ORIGINS=http://localhost:8001,http://127.0.0.1:8001`
- `ALLOWED_HOSTS=localhost,127.0.0.1`
- `RATE_LIMIT_WINDOW_SECONDS=60`
- `RATE_LIMIT_MAX_REQUESTS=30`

---

## 11) Error Handling Strategy

- User-facing API errors are generic and safe.
- Internal details are logged server-side.
- LLM failures degrade gracefully with fallback prompt.
- Startup failures surface early (dataset/model/index initialization).

---

## 12) Tools Included in Repository

- `tools/load_dataset.py`
  - downloads and validates HS dataset.
- `tools/build_index.py`
  - builds index and runs test queries for quality checks.

---

## 13) Production Readiness Checklist

- [x] No hardcoded API keys in source
- [x] `.env` excluded from git
- [x] Input validation + sanitization
- [x] Rate limiting
- [x] CORS/host restrictions configurable
- [x] Sensitive logging reduced
- [x] Graceful LLM failure fallback

---

## 14) Suggested Next Improvements (Optional)

- Move rate limit storage to Redis for multi-instance deployment.
- Add request tracing IDs for observability.
- Add CI dependency vulnerability scans (`pip-audit`).
- Add automated API tests for security behavior (429, malformed payloads, host/origin checks).
