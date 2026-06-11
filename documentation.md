# HS Code Finder — Product & Flow Documentation

## 1) Product Overview

HS Code Finder is an AI-assisted classification tool that helps users identify the correct **6-digit Harmonized System (HS) code** for a product.

### Core value

- Accept plain-language product descriptions.
- Retrieve best HS candidates using semantic search.
- Use LLM reasoning to ask clarifying questions.
- Return final HS code with explanation.

---

## 2) End-to-End User Flow

1. User opens the web app at `/`.
2. User types a product description (example: "frozen shrimp seafood").
3. Frontend sends request to `POST /api/classify`.
4. Backend validates and sanitizes input.
5. Classification engine either:
   - starts a new session (if `session_id` is null), or
   - continues existing session.
6. System creates embeddings for user text.
7. FAISS vector search finds top HS candidates.
8. OpenAI receives candidate context + conversation context.
9. OpenAI returns either:
   - a short clarifying question, or
   - a final classification in structured format.
10. Backend parses response and returns JSON to frontend.
11. Frontend renders either a follow-up question or a final result card with hierarchy path.

---

## 3) Backend Runtime Flow (Startup)

On application startup (`backend/main.py` lifespan):

1. Load `.env` variables.
2. Load HS dataset (`data/hs_codes.csv`) into memory.
3. Load embedding model (`all-MiniLM-L6-v2`).
4. Build FAISS index from 6-digit subheadings only, with hierarchy-enriched text (chapter → heading → subheading descriptions concatenated for richer semantic context).
5. Initialize OpenAI client.
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
2. Request model validates input.
3. Classification engine processes query.
4. Returns `ClassifyResponse` with type `question` or `result`.

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
- Send candidates to OpenAI.
- Parse OpenAI output:
  - if `RESULT:`, build final result
  - else return clarifying question

### Existing session

- Append user follow-up answer.
- Reuse original candidate set (re-searching with follow-up answers corrupts candidates).
- Send updated conversation history to OpenAI.
- Parse and return question/result.
- After MAX_QUESTIONS answers are processed, force the top candidate as the result.

### Safeguards

- Empty message protection.
- Max question cap (`MAX_QUESTIONS=5`) — the user's final answer is always processed before forcing.
- REFINE signal: if the LLM detects wrong candidates, automatic retry with better search terms (one retry max).
- Session eviction: completed sessions are deleted immediately; active sessions use LRU eviction at 1000.
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

- Uses OpenAI GPT-4o Mini (configurable via `OPENAI_MODEL` env var) via the `openai` SDK.
- System prompt enforces 5-step classification: relevance check → normalization → product profile → smart questioning → confidence gating.
- `expand_query()`: translates consumer language to HS trade terminology for retrieval (results cached in memory).
- 30-second timeout on all OpenAI requests.
- Single API call per turn; returns a fallback message on error.

---

## 7) Data Layer

### Source

- `data/hs_codes.csv`
- Includes section, code, description, parent, level.

### Knowledge Base (`backend/services/hs_knowledge.py`)

- Loads CSV and validates schema.
- Stores indexed structures by code and by parent.
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

- API key is loaded from environment (`OPENAI_API_KEY`).
- `.env` is ignored in VCS.
- `.env.example` is template-only.

### Request safety

- Pydantic schema validation (control-char cleanup, whitespace normalization, 1000-char cap).
- Rate limiting: 20 requests/minute per IP on `/api/classify` (via slowapi).
- Security headers on all responses: `X-Frame-Options`, `X-Content-Type-Options`, `Referrer-Policy`, `X-XSS-Protection`.
- Prompt injection hardening: user text is wrapped in XML tags (`<product>`, `<answer>`) and treated as data, not instructions.

---

## 10) Environment Variables

Required:

- `OPENAI_API_KEY`

Optional:

- `OPENAI_MODEL` (default: `gpt-4o-mini`) — override the OpenAI model used for classification.
- `SERVE_FRONTEND=true` (default: `true`)
  - `true` — FastAPI serves both frontend and API (local dev)
  - `false` — backend API only (Render backend + Vercel frontend)

---

## 11) Error Handling Strategy

- User-facing API errors are generic and safe.
- Internal details are logged server-side.
- LLM failures degrade gracefully with a fallback prompt asking for more detail.
- Startup failures surface early (dataset/model/index initialization).

---

## 12) Tools Included in Repository

- `tools/load_dataset.py` — downloads and validates the HS dataset.
- `tools/build_index.py` — builds index and runs test queries for quality checks.
