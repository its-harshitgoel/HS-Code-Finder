import re
import uuid
from collections import OrderedDict

from models.schemas import Candidate, ClassifyResponse, FinalResult, SessionState
from services.embedding import EmbeddingService
from services.hs_knowledge import HSKnowledgeBase
from services.llm_service import OpenAIService
from services.vector_search import VectorSearchService
from utils.logger import get_logger

logger = get_logger("classifier")

TOP_K_CANDIDATES = 10
MAX_QUESTIONS = 5
MAX_SESSIONS = 1000

# Matches LLM step-header preamble that sometimes appears before the actual question.
# Used to strip content like "STEP 4 — SMART QUESTIONING: " that leaks from chain-of-thought.
_LLM_PREAMBLE = re.compile(
    r"^\s*(?:(?:══\s*)?step\s+\d+[^:?]*:[   ]*|smart\s+question(?:ing)?[^:?]*:[   ]*)",
    re.IGNORECASE,
)


def _simplify_query(description: str) -> str:
    """Strip capability/feature language that misleads the embedding model.

    'metal water bottle that can keep temperature for 12 hours' → 'metal water bottle'
    """
    text = description.strip()
    text = re.sub(
        r"\s+that\s+(can|will|could|would|is|are|does|keeps?|maintains?|provides?|allows?)\b.*",
        "", text, flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+which\s+(can|will|could|would|is|are)\b.*", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\s+(designed to|made to|used to|capable of|able to)\b.*", "", text, flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+(for over|for up to|up to|over)\s+\d+.*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip() or description


class ClassificationEngine:
    def __init__(
        self,
        knowledge_base: HSKnowledgeBase,
        embedding_service: EmbeddingService,
        vector_search: VectorSearchService,
        llm_service: OpenAIService,
    ) -> None:
        self._kb = knowledge_base
        self._embed = embedding_service
        self._search = vector_search
        self._llm = llm_service
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()

    # ──────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────

    def classify(self, session_id: str | None, message: str) -> ClassifyResponse:
        message = message.strip()
        if not message:
            return self._error_response("Please describe the product you'd like to classify.")

        if session_id is None or session_id not in self._sessions:
            return self._start_new_session(message)

        self._sessions.move_to_end(session_id)
        return self._process_answer(session_id, message)

    # ──────────────────────────────────────────────
    # Session lifecycle
    # ──────────────────────────────────────────────

    def _start_new_session(self, description: str) -> ClassifyResponse:
        session_id = str(uuid.uuid4())
        logger.info("New session %s (len=%d)", session_id[:8], len(description))

        session = SessionState(session_id=session_id, original_query=description)

        try:
            candidates = self._retrieve_candidates(description)
        except Exception as e:
            logger.error("Search failed: %s", e)
            return self._fallback_response(session_id)

        if not candidates:
            return self._fallback_response(session_id)

        session.candidates = candidates
        if len(self._sessions) >= MAX_SESSIONS:
            self._sessions.popitem(last=False)
        self._sessions[session_id] = session

        user_msg = self._first_message(description, candidates)
        llm_response = self._llm.generate_response([user_msg])
        session.llm_history = [user_msg, {"role": "assistant", "content": llm_response}]

        result = self._parse_llm_response(session, llm_response)
        self._clean_last_history_entry(session, result)
        return result

    def _process_answer(self, session_id: str, answer: str) -> ClassifyResponse:
        session = self._sessions.get(session_id)
        if session is None:
            logger.warning("Session %s evicted mid-conversation, restarting", session_id[:8])
            return self._start_new_session(answer)

        # ── Restart path: user was asked to rephrase ──
        if session.awaiting_product_description:
            session.awaiting_product_description = False
            session.original_query = answer
            session.candidates = []
            session.questions_asked = 0
            session.llm_history = []
            session.confirmed_qa = []
            session.retrieval_retried = False
            session.enriched_retrieval_done = False
            try:
                candidates = self._retrieve_candidates(answer)
            except Exception as e:
                logger.error("Search failed on restart: %s", e)
                return self._fallback_response(session_id)
            if not candidates:
                return self._fallback_response(session_id)
            session.candidates = candidates
            user_msg = self._first_message(answer, candidates)
            llm_response = self._llm.generate_response([user_msg])
            session.llm_history = [user_msg, {"role": "assistant", "content": llm_response}]
            result = self._parse_llm_response(session, llm_response)
            self._clean_last_history_entry(session, result)
            return result

        # ── FIX Issue 1: record Q→A pair BEFORE building the follow-up message ──
        # The last assistant message in history is the cleaned question we just asked.
        # Pairing it with the user's answer gives the LLM an explicit confirmed-facts block.
        if session.llm_history and session.llm_history[-1]["role"] == "assistant":
            prev_question = session.llm_history[-1]["content"]
            session.confirmed_qa.append({"q": prev_question, "a": answer})

        session.questions_asked += 1

        user_msg = self._followup_message(answer, session.candidates, session)
        llm_response = self._llm.generate_response(session.llm_history + [user_msg])
        session.llm_history.append(user_msg)
        session.llm_history.append({"role": "assistant", "content": llm_response})

        result = self._parse_llm_response(session, llm_response)
        self._clean_last_history_entry(session, result)

        # Always process the user's answer; only force a result if the LLM is STILL asking
        if result.type == "question" and session.questions_asked >= MAX_QUESTIONS:
            logger.info("Max questions reached for session %s, forcing result", session_id[:8])
            return self._force_result(session, session.candidates[0])

        return result

    # ──────────────────────────────────────────────
    # LLM response parsing
    # ──────────────────────────────────────────────

    def _parse_llm_response(self, session: SessionState, llm_text: str) -> ClassifyResponse:
        result_match = re.search(r"RESULT:\s*(\d{6})", llm_text, re.IGNORECASE)
        if result_match:
            return self._build_result(session, result_match.group(1), llm_text)

        refine_match = re.search(r"^REFINE:\s*(.+?)$", llm_text, re.IGNORECASE | re.MULTILINE)
        if refine_match:
            return self._handle_refine(session, refine_match.group(1).strip())

        if re.search(r"RESULT:", llm_text, re.IGNORECASE):
            return self._handle_refine(session, "")

        text = re.sub(r"^(QUESTION|Q)\s*:\s*", "", llm_text.strip(), flags=re.IGNORECASE).strip()

        # Extract OPTIONS before stripping
        options: list[str] | None = None
        options_match = re.search(r"\nOPTIONS:\s*(.+)", text, re.IGNORECASE)
        if options_match:
            raw = [o.strip() for o in options_match.group(1).split("|") if o.strip()]
            if len(raw) >= 2:
                options = raw
            text = re.sub(r"\nOPTIONS:.*", "", text, flags=re.IGNORECASE | re.DOTALL).strip()

        # ── FIX Issue 3: line-by-line extraction instead of double-newline paragraph split ──
        # Scanning lines in reverse finds the actual question even when step headers are
        # on the immediately preceding line (single \n), which paragraph splitting misses.
        # _LLM_PREAMBLE strips inline header prefixes like "STEP 4 — SMART QUESTIONING: ".
        if "?" in text:
            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            for ln in reversed(lines):
                candidate = _LLM_PREAMBLE.sub("", ln).strip()
                if not candidate.endswith("?"):
                    candidate = ln  # regex stripped too much; fall back to raw line
                if candidate.endswith("?"):
                    last_dot = candidate.rfind(". ")
                    last_excl = candidate.rfind("! ")
                    brk = max(last_dot, last_excl)
                    text = candidate[brk + 2:].strip() if brk != -1 else candidate
                    break

        if not text.endswith("?"):
            session.awaiting_product_description = True

        return ClassifyResponse(
            session_id=session.session_id,
            type="question",
            message=text,
            candidates=[],
            options=options,
        )

    # ──────────────────────────────────────────────
    # Result construction
    # ──────────────────────────────────────────────

    def _build_result(self, session: SessionState, hs_code: str, llm_text: str) -> ClassifyResponse:
        entry = self._kb.get_by_code(hs_code)
        entry_desc = (
            entry.description if entry
            else next(
                (c.description for c in session.candidates if c.hs_code == hs_code),
                "Classification result",
            )
        )

        explanation_match = re.search(
            r"EXPLANATION:\s*(.+?)(?:\n|$)", llm_text, re.IGNORECASE | re.DOTALL
        )
        explanation = explanation_match.group(1).strip() if explanation_match else ""

        hierarchy = self._kb.get_hierarchy_path(hs_code)
        if hierarchy:
            hierarchy_text = " → ".join(f"{e.hs_code}: {e.description}" for e in hierarchy)
            explanation += f"\n\n**Classification path:** {hierarchy_text}"

        message = f"**HS Code {hs_code}**\n\n**{entry_desc}**\n\n{explanation}"
        logger.info("Session %s result: %s", session.session_id[:8], hs_code)

        self._sessions.pop(session.session_id, None)

        return ClassifyResponse(
            session_id=session.session_id,
            type="result",
            message=message,
            candidates=[c for c in session.candidates if c.hs_code == hs_code][:1]
            or session.candidates[:1],
            final_result=FinalResult(
                hs_code=hs_code, description=entry_desc, explanation=explanation
            ),
        )

    def _force_result(self, session: SessionState, candidate: Candidate) -> ClassifyResponse:
        hierarchy = self._kb.get_hierarchy_path(candidate.hs_code)
        explanation = "Based on your description and answers, this is the best matching code."
        if hierarchy:
            hierarchy_text = " → ".join(f"{e.hs_code}: {e.description}" for e in hierarchy)
            explanation += f"\n\n**Classification path:** {hierarchy_text}"

        message = f"**HS Code {candidate.hs_code}**\n\n**{candidate.description}**\n\n{explanation}"

        self._sessions.pop(session.session_id, None)

        return ClassifyResponse(
            session_id=session.session_id,
            type="result",
            message=message,
            candidates=[candidate],
            final_result=FinalResult(
                hs_code=candidate.hs_code,
                description=candidate.description,
                explanation=explanation,
            ),
        )

    # ──────────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────────

    def _handle_refine(self, session: SessionState, refined_terms: str) -> ClassifyResponse:
        """Two-stage REFINE retry.

        Stage 1 (retrieval_retried=False): use LLM-suggested terms or simplified query.
        Stage 2 (enriched_retrieval_done=False, Q&A available): enrich with accumulated
          user answers — bridges the gap when initial retrieval was vague and the LLM
          collected enough context to search more precisely.
        """
        can_enrich = bool(session.confirmed_qa) and not session.enriched_retrieval_done

        if not session.retrieval_retried:
            session.retrieval_retried = True
            query = refined_terms or _simplify_query(session.original_query)
            logger.info("REFINE stage 1 — query: %r", query)
        elif can_enrich:
            # ── FIX Issue 2 ──
            # Build an enriched query from all facts the user has provided so far.
            # This is the key missing path: initial retrieval was vague ("smart water bottle")
            # but after Q&A we know it's "insulated stainless steel vacuum flask" — search that.
            session.enriched_retrieval_done = True
            answers = " ".join(qa["a"] for qa in session.confirmed_qa)
            query = f"{session.original_query} {answers}".strip()
            logger.info("REFINE stage 2 (enriched) — query: %r", query[:80])
        else:
            session.awaiting_product_description = True
            return ClassifyResponse(
                session_id=session.session_id,
                type="question",
                message=(
                    "I couldn't find relevant HS candidates for your product. "
                    "Could you describe it more specifically — what it's made of, "
                    "what it does, and what category it belongs to?"
                ),
                candidates=[],
            )

        try:
            new_candidates = self._retrieve_candidates(query)
        except Exception as e:
            logger.error("Fallback retrieval failed: %s", e)
            new_candidates = []

        if new_candidates:
            session.candidates = new_candidates
            user_msg = self._first_message(session.original_query, new_candidates)
            llm_response = self._llm.generate_response([user_msg])
            session.llm_history = [user_msg, {"role": "assistant", "content": llm_response}]
            result = self._parse_llm_response(session, llm_response)
            self._clean_last_history_entry(session, result)
            return result

        # This stage's retrieval failed — try next stage on next REFINE signal
        session.awaiting_product_description = True
        return ClassifyResponse(
            session_id=session.session_id,
            type="question",
            message=(
                "I couldn't find relevant HS candidates for your product. "
                "Could you describe it more specifically — what it's made of, "
                "what it does, and what category it belongs to?"
            ),
            candidates=[],
        )

    def _retrieve_candidates(self, description: str) -> list[Candidate]:
        """Three-pass retrieval bridging consumer language → HS terminology.

        Pass 1: raw description
        Pass 2: simplified (strips capability/feature language)
        Pass 3: LLM-generated HS trade terms (cached per description)
        """
        seen: dict[str, Candidate] = {}

        def _merge(candidates: list[Candidate]) -> None:
            for c in candidates:
                if c.hs_code not in seen or c.similarity_score > seen[c.hs_code].similarity_score:
                    seen[c.hs_code] = c

        _merge(self._search.search(self._embed.encode(description), top_k=TOP_K_CANDIDATES))

        simplified = _simplify_query(description)
        if simplified.lower() != description.lower():
            logger.info("Pass 2 — simplified: %r", simplified)
            _merge(self._search.search(self._embed.encode(simplified), top_k=TOP_K_CANDIDATES))

        trade_terms = self._llm.expand_query(description)
        logger.info("Pass 3 — trade terms: %s", trade_terms)
        for term in trade_terms:
            _merge(self._search.search(self._embed.encode(term), top_k=8))

        return sorted(seen.values(), key=lambda c: c.similarity_score, reverse=True)[:TOP_K_CANDIDATES]

    # ──────────────────────────────────────────────
    # Message builders
    # ──────────────────────────────────────────────

    def _first_message(self, description: str, candidates: list[Candidate]) -> dict:
        return {
            "role": "user",
            "content": (
                f"Classify this product: <product>{description}</product>\n\n"
                f"Candidate HS codes:\n{self._format_candidates(candidates)}\n\n"
                "Normalize the product description, build a product profile from known attributes, "
                "then ask the highest-value clarifying question. "
                "Only classify when no further question can change the outcome."
            ),
        }

    def _followup_message(
        self, answer: str, candidates: list[Candidate], session: SessionState
    ) -> dict:
        # ── FIX Issue 1: include explicit Q→A pairs so the LLM can't re-ask them ──
        # confirmed_qa already includes THIS turn's answer (recorded in _process_answer).
        if session.confirmed_qa:
            pairs = "\n".join(f"  • {qa['q']} → {qa['a']}" for qa in session.confirmed_qa)
            facts_block = f"Confirmed from conversation (do NOT ask about these again):\n{pairs}\n\n"
        else:
            facts_block = ""

        return {
            "role": "user",
            "content": (
                f"User's answer: <answer>{answer}</answer>\n\n"
                f"{facts_block}"
                f"Original product description: {session.original_query}\n\n"
                f"Candidate HS codes:\n{self._format_candidates(candidates)}\n\n"
                "Update the product profile with this answer. "
                "If you can now classify, output RESULT. "
                "Otherwise ask the ONE remaining unknown attribute that would change the heading."
            ),
        }

    def _clean_last_history_entry(self, session: SessionState, result: ClassifyResponse) -> None:
        """Replace the last assistant history entry with only the question text.

        Storing chain-of-thought verbatim means the LLM re-reads hundreds of words of its
        own reasoning on every subsequent turn and loses track of what it already asked.
        Storing only the clean question keeps context concise and prevents repeated questions.
        """
        if result.type == "question" and result.message and session.llm_history:
            session.llm_history[-1] = {"role": "assistant", "content": result.message}

    def _format_candidates(self, candidates: list[Candidate]) -> str:
        return "\n".join(
            f"{i}. HS {c.hs_code} — {c.description} (similarity: {c.similarity_score:.0%}, level: {c.level})"
            for i, c in enumerate(candidates, 1)
        )

    # ──────────────────────────────────────────────
    # Error / fallback responses
    # ──────────────────────────────────────────────

    def _fallback_response(self, session_id: str) -> ClassifyResponse:
        return ClassifyResponse(
            session_id=session_id,
            type="question",
            message=(
                "I couldn't find a close match. Could you describe the product "
                "differently? Mention what it's made of, what it's used for, "
                "or what category it might fall into."
            ),
            candidates=[],
        )

    def _error_response(self, message: str) -> ClassifyResponse:
        return ClassifyResponse(
            session_id=str(uuid.uuid4()),
            type="question",
            message=message,
            candidates=[],
        )
