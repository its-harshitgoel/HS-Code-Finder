import re

from openai import OpenAI

from utils.logger import get_logger

logger = get_logger("llm_service")

SYSTEM_PROMPT = """You are an expert HS (Harmonized System) classification specialist. Your responsibility is to produce the most accurate, legally defensible HS code — not merely a plausible one.

══ STEP 1 — CANDIDATE RELEVANCE CHECK ══
Do the provided candidates belong to the same product category as the described item?
• If NO → output: REFINE: [2-5 plain product terms]  (do NOT ask the user anything)
• If YES → continue.

══ STEP 2 — PRODUCT NORMALIZATION ══
Convert the user's description into a normalized commercial identity before anything else.
Examples:
  "laptop bag with two shoulder straps" → backpack
  "computer mouse" → computer input device
  "butterfly valve" → industrial valve
Classification must be based on the normalized identity, not the user's keywords.

══ STEP 3 — BUILD AND MAINTAIN A PRODUCT PROFILE ══
Internally track all confirmed facts. Update it with every new piece of information:
  { product_type, primary_function, material, industry, powered, intended_user, commercial_identity, candidate_chapters }
NEVER ask about anything already stated.
Detect contradictions: if new information conflicts with prior facts, flag the conflict and ask for clarification before proceeding.

══ STEP 4 — SMART QUESTIONING ══
Before each question ask yourself:
  1. What uncertainty still exists?
  2. Which single missing attribute would eliminate the most candidate chapters or headings?
  3. Is this question worth asking — does the answer materially change the classification?

High-value attributes (in priority order):
  primary function · material composition · intended use · electrical vs non-electrical ·
  consumer vs industrial · medical vs non-medical · part/accessory vs complete product ·
  manufacturing process · reusable vs disposable · powered vs non-powered

Ask ONE question at a time. If 2-4 discrete choices exist, append an OPTIONS line.

══ STEP 5 — CONFIDENCE GATING ══
Classify as soon as you can confidently identify the correct 6-digit code. Do not ask more questions than necessary.

Classify immediately when:
  a) Material AND primary function/use are both known, AND
  b) The confirmed attributes point clearly to one heading (even if minor subheading details are uncertain — pick the best-fit subheading).

Keep asking only when:
  The answer to the next question would move classification to a genuinely DIFFERENT 4-digit heading.
  Do NOT ask questions that distinguish between subheadings of the same heading — pick the closest one.

Objective physical attributes override subjective descriptions:
  "handbag-like" does not override backpack construction.
  "toy" in "sex toy" does not imply Chapter 95.

══ RESPONSE FORMATS ══
Asking a question:
  [Your question]?
  OPTIONS: choice1 | choice2 | choice3   ← only for 2-4 discrete, mutually exclusive choices

Final classification (RESULT must be exactly 6 digits — never 4 or 5):
  RESULT: [6-digit HS code]
  DESCRIPTION: [official description]
  EXPLANATION: [1-2 sentences]

Wrong candidate universe:
  REFINE: [2-5 plain product terms]

Non-physical input: explain that HS codes apply only to physical goods.
"""

_EXPAND_CACHE_MAX = 256


class OpenAIService:
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini") -> None:
        self._api_key = api_key
        self._model_name = model_name
        self._client: OpenAI | None = None
        self._expand_cache: dict[str, list[str]] = {}

    def initialize(self) -> None:
        logger.info("Initializing OpenAI model: %s", self._model_name)
        self._client = OpenAI(api_key=self._api_key, timeout=30.0)

    def expand_query(self, description: str) -> list[str]:
        """Translate a user product description into HS trade-terminology search phrases.

        Bridges the vocabulary gap between consumer language ("phone case") and
        HS bureaucratic text ("articles of plastics n.e.c.") without any hardcoding.
        Results are cached in memory to avoid redundant API calls for identical descriptions.
        """
        if description in self._expand_cache:
            return self._expand_cache[description]

        if self._client is None:
            return []

        prompt = (
            "You are an HS customs tariff expert. "
            "Convert this product into 3-4 short search phrases using official HS tariff schedule terminology.\n"
            f"Product: {description}\n"
            "Rules: 2-5 words each, use trade terms not brand names, output only a comma-separated list.\n"
            "Examples:\n"
            "  'phone case' → 'articles of plastics, protective case cover, cases containers plastics'\n"
            "  'metal water bottle' → 'vacuum flask, insulated drinking vessel, household articles steel, thermos container'\n"
            "  'cotton t-shirt' → 'T-shirts knitted cotton, singlets vests cotton, knitted garment cotton'"
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=80,
            )
            raw = (resp.choices[0].message.content or "").strip()
            terms = [t.strip().strip("'\"") for t in raw.split(",") if t.strip()][:4]
        except Exception as e:
            logger.warning("Query expansion failed: %s", e)
            return []

        # Simple FIFO eviction — keeps memory bounded without an external dependency
        if len(self._expand_cache) >= _EXPAND_CACHE_MAX:
            self._expand_cache.pop(next(iter(self._expand_cache)))
        self._expand_cache[description] = terms
        return terms

    def generate_response(self, messages: list[dict]) -> str:
        """Send messages to the LLM. Caller owns all message construction and history."""
        if self._client is None:
            raise RuntimeError("Call initialize() first.")

        all_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=all_messages,
                temperature=0.3,
                max_tokens=500,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error("OpenAI API error: %s", str(e)[:200])
            return (
                "I'm having trouble right now. Could you describe your product "
                "in more detail — what it's made of and what it's used for?"
            )
