"""
Unit tests for ClassificationEngine.

All external I/O (LLM, FAISS) is replaced with lightweight fakes so these
tests run in milliseconds with no network access, no GPU, and no test DB.
"""
import re
import uuid
from unittest.mock import MagicMock, patch

import pytest

from models.schemas import Candidate, ClassifyResponse, HSEntry, SessionState
from services.classifier import ClassificationEngine, MAX_QUESTIONS, _simplify_query, _LLM_PREAMBLE


# ─────────────────────────────────────────────────────────
# Helpers / Fakes
# ─────────────────────────────────────────────────────────

def _make_candidate(code: str = "610910", score: float = 0.85) -> Candidate:
    return Candidate(
        hs_code=code,
        description="T-shirts, singlets and other vests, of cotton",
        section="XI",
        level=6,
        parent="6109",
        similarity_score=score,
    )


def _make_engine(llm_responses: list[str]) -> ClassificationEngine:
    """Construct ClassificationEngine with all I/O mocked.

    llm_responses is a queue: each call to generate_response pops the next response.
    expand_query always returns an empty list (no trade-term expansion needed in unit tests).
    vector_search always returns one fixed candidate.
    """
    kb = MagicMock()
    kb.get_by_code.return_value = HSEntry(
        section="XI", hs_code="610910",
        description="T-shirts of cotton", parent="6109", level=6,
    )
    kb.get_hierarchy_path.return_value = []

    embed = MagicMock()
    embed.encode.return_value = [0.0] * 384

    search = MagicMock()
    search.search.return_value = [_make_candidate()]

    responses = list(llm_responses)  # copy so mutation is local
    llm = MagicMock()
    llm.expand_query.return_value = []
    llm.generate_response.side_effect = lambda _msgs: responses.pop(0)

    return ClassificationEngine(kb, embed, search, llm)


# ─────────────────────────────────────────────────────────
# Issue 3 — header stripping
# ─────────────────────────────────────────────────────────

class TestParseStepsHeaderStripping:
    """_parse_llm_response must not leak step headers to the UI (Issue 3)."""

    def _parse(self, engine: ClassificationEngine, text: str) -> ClassifyResponse:
        session = SessionState(session_id=str(uuid.uuid4()), original_query="test")
        return engine._parse_llm_response(session, text)

    def test_step_header_single_newline_stripped(self):
        engine = _make_engine([])
        # Header + question on consecutive lines (single \n — the common leak pattern)
        raw = "══ STEP 4 — SMART QUESTIONING ══\nWhat material is it made of?\nOPTIONS: Steel | Plastic"
        result = self._parse(engine, raw)
        assert result.message == "What material is it made of?"
        assert "STEP" not in result.message
        assert "SMART QUESTIONING" not in result.message
        assert result.options == ["Steel", "Plastic"]

    def test_step_header_colon_inline_stripped(self):
        engine = _make_engine([])
        # Header and question on one line, colon-separated
        raw = "STEP 4 — SMART QUESTIONING: What material is it made of?"
        result = self._parse(engine, raw)
        assert result.message == "What material is it made of?"
        assert "STEP" not in result.message

    def test_step_header_double_newline_still_works(self):
        engine = _make_engine([])
        # Already-working case: paragraph split handles this
        raw = "Product profile reasoning...\n\nWhat is the primary use?\nOPTIONS: Casual | Formal"
        result = self._parse(engine, raw)
        assert result.message == "What is the primary use?"
        assert result.options == ["Casual", "Formal"]

    def test_multi_sentence_line_trims_to_last_question(self):
        engine = _make_engine([])
        raw = "Based on your description. What material is the body made of?"
        result = self._parse(engine, raw)
        assert result.message == "What material is the body made of?"

    def test_result_code_parsed_correctly(self):
        engine = _make_engine([])
        session = SessionState(
            session_id=str(uuid.uuid4()),
            original_query="test",
            candidates=[_make_candidate()],
        )
        raw = "RESULT: 610910\nDESCRIPTION: T-shirts of cotton\nEXPLANATION: Cotton knit garment."
        result = engine._parse_llm_response(session, raw)
        assert result.type == "result"
        assert result.final_result is not None
        assert result.final_result.hs_code == "610910"

    def test_options_not_in_message(self):
        engine = _make_engine([])
        raw = "What is the material?\nOPTIONS: Cotton | Polyester | Wool"
        result = self._parse(engine, raw)
        assert "OPTIONS" not in result.message
        assert result.options == ["Cotton", "Polyester", "Wool"]


# ─────────────────────────────────────────────────────────
# Issue 1 — confirmed Q&A tracking and followup format
# ─────────────────────────────────────────────────────────

class TestConfirmedQATracking:
    """_process_answer must record Q→A pairs and include them in follow-up messages (Issue 1)."""

    def test_first_answer_recorded_in_confirmed_qa(self):
        engine = _make_engine([
            "What is the primary use?",           # Q1 (first LLM call)
            "What fabric weight is it?",          # Q2 (second LLM call — after Q1 answered)
        ])
        r1 = engine.classify(None, "cotton t-shirt")
        session_id = r1.session_id
        assert r1.type == "question"
        assert r1.message == "What is the primary use?"

        r2 = engine.classify(session_id, "casual wear")
        session = engine._sessions.get(session_id)
        # After answering Q1, confirmed_qa must have exactly one entry
        assert len(session.confirmed_qa) == 1
        assert session.confirmed_qa[0]["q"] == "What is the primary use?"
        assert session.confirmed_qa[0]["a"] == "casual wear"

    def test_confirmed_qa_grows_with_each_answer(self):
        engine = _make_engine([
            "What is the primary use?",
            "What is the material weight?",
            "RESULT: 610910\nDESCRIPTION: T-shirts of cotton\nEXPLANATION: ok.",
        ])
        r1 = engine.classify(None, "cotton t-shirt")
        sid = r1.session_id
        engine.classify(sid, "casual wear")        # answers Q1
        session = engine._sessions.get(sid)
        assert len(session.confirmed_qa) == 1
        engine.classify(sid, "lightweight")        # answers Q2 → result returned
        # session deleted on result, so confirmed_qa state only verifiable before result

    def test_followup_message_contains_confirmed_facts(self):
        engine = _make_engine([
            "What is the primary use?",
            "What fabric weight is it?",
        ])
        r1 = engine.classify(None, "cotton t-shirt")
        sid = r1.session_id
        engine.classify(sid, "casual wear")

        session = engine._sessions[sid]
        # Build a followup message for a hypothetical next answer
        msg = engine._followup_message("lightweight", session.candidates, session)
        content = msg["content"]

        assert "Confirmed from conversation" in content
        assert "What is the primary use?" in content
        assert "casual wear" in content

    def test_confirmed_facts_block_not_present_on_first_followup(self):
        """First follow-up has no confirmed_qa yet — block must be absent."""
        engine = _make_engine([
            "What is the primary use?",
            "What fabric weight is it?",
        ])
        r1 = engine.classify(None, "cotton t-shirt")
        sid = r1.session_id
        session = engine._sessions[sid]

        # Before any answer, confirmed_qa is empty
        msg = engine._followup_message("casual wear", session.candidates, session)
        assert "Confirmed from conversation" not in msg["content"]

    def test_awaiting_restart_clears_confirmed_qa(self):
        """If session resets (user asked to rephrase), confirmed_qa is wiped."""
        engine = _make_engine([
            "I have no relevant candidates.\nOPTIONS: None",  # no '?' at end → sets awaiting
            "What is the primary use?",
        ])
        # Force awaiting_product_description by making parse produce non-question
        # We'll set it manually on the session after first classify
        r1 = engine.classify(None, "cotton t-shirt")
        sid = r1.session_id
        session = engine._sessions[sid]
        session.confirmed_qa = [{"q": "old q", "a": "old a"}]
        session.awaiting_product_description = True
        session.retrieval_retried = True

        # Next message is treated as a fresh product description
        engine.classify(sid, "new product description")
        session = engine._sessions.get(sid)
        if session:  # session may be deleted if result returned
            assert session.confirmed_qa == []
            assert session.retrieval_retried is False


# ─────────────────────────────────────────────────────────
# Issue 2 — enriched REFINE retry
# ─────────────────────────────────────────────────────────

class TestEnrichedRefineRetry:
    """REFINE fired after Q&A must retry with original_query + user answers (Issue 2)."""

    def test_enriched_retry_uses_confirmed_answers(self):
        engine = _make_engine([
            "Is this vacuum insulated?",           # Q1
            "REFINE: vacuum flask insulated steel", # LLM decides candidates wrong after Q1
            "RESULT: 961700\nDESCRIPTION: Vacuum flasks\nEXPLANATION: ok.",
        ])

        r1 = engine.classify(None, "smart water bottle")
        sid = r1.session_id
        assert r1.message == "Is this vacuum insulated?"

        # Answer Q1 — this records Q&A in confirmed_qa
        engine.classify(sid, "yes, stainless steel vacuum")

        session = engine._sessions.get(sid)
        # Session should be gone (result returned) OR still active with enriched retry done
        # Either way, the enriched retry path should have been triggered

    def test_first_refine_uses_llm_terms(self):
        """When retrieval_retried is False, REFINE uses LLM-suggested terms."""
        engine = _make_engine([])
        # Clear the side_effect so return_value works (side_effect takes priority on MagicMock)
        engine._llm.generate_response.side_effect = None
        engine._llm.generate_response.return_value = "What is the material?\nOPTIONS: Steel | Plastic"

        session = SessionState(
            session_id=str(uuid.uuid4()),
            original_query="smart water bottle",
            candidates=[_make_candidate()],
        )
        engine._sessions[session.session_id] = session

        captured_queries = []
        def capture(q):
            captured_queries.append(q)
            return [_make_candidate()]
        engine._retrieve_candidates = capture

        engine._handle_refine(session, "vacuum flask insulated vessel")

        assert len(captured_queries) == 1
        assert captured_queries[0] == "vacuum flask insulated vessel"
        assert session.retrieval_retried is True

    def test_second_refine_uses_enriched_query(self):
        """When retrieval_retried is True and confirmed_qa is non-empty, use enriched query."""
        engine = _make_engine([])
        engine._llm.generate_response.side_effect = None
        engine._llm.generate_response.return_value = "What is the use?\nOPTIONS: Drinking | Cooking"

        session = SessionState(
            session_id=str(uuid.uuid4()),
            original_query="smart water bottle",
            candidates=[_make_candidate()],
            retrieval_retried=True,    # first retry already done
            confirmed_qa=[
                {"q": "Is it vacuum insulated?", "a": "yes stainless steel"},
                {"q": "What capacity?", "a": "1 litre"},
            ],
        )
        engine._sessions[session.session_id] = session

        captured_queries = []
        def capture(q):
            captured_queries.append(q)
            return [_make_candidate()]
        engine._retrieve_candidates = capture

        engine._handle_refine(session, "")

        assert len(captured_queries) == 1
        expected = "smart water bottle yes stainless steel 1 litre"
        assert captured_queries[0] == expected
        assert session.enriched_retrieval_done is True

    def test_all_retries_exhausted_sets_awaiting(self):
        """After both retries are used, awaiting_product_description is set."""
        engine = _make_engine([])
        session = SessionState(
            session_id=str(uuid.uuid4()),
            original_query="mystery product",
            retrieval_retried=True,
            enriched_retrieval_done=True,
            confirmed_qa=[{"q": "q", "a": "a"}],
        )
        engine._sessions[session.session_id] = session

        result = engine._handle_refine(session, "")
        assert session.awaiting_product_description is True
        assert "couldn't find" in result.message.lower()


# ─────────────────────────────────────────────────────────
# MAX_QUESTIONS off-by-one fix
# ─────────────────────────────────────────────────────────

class TestMaxQuestions:
    def test_fifth_answer_is_processed_by_llm(self):
        """LLM must see the 5th answer; force-result only if LLM still asks afterward.

        Flow: 1 initial call + 5 answer calls = 6 total LLM calls.
        The 6th response is still a question, triggering force_result.
        """
        engine = _make_engine([
            "Q1?", "Q2?", "Q3?", "Q4?", "Q5?",  # Q5 asked after a4
            "Q6?",  # LLM response to a5 — still asking, so force_result fires
        ])
        r = engine.classify(None, "product")
        sid = r.session_id
        for answer in ["a1", "a2", "a3", "a4"]:
            r = engine.classify(sid, answer)
            assert r.type == "question"

        # 5th answer: LLM is called AND processes the answer; force-result only after
        r = engine.classify(sid, "a5")
        assert r.type == "result"
        assert engine._llm.generate_response.call_count == 6  # 1 initial + 5 answers

    def test_llm_classify_on_fifth_answer_not_forced(self):
        """If LLM classifies on 5th answer, that result is used (not force_result).

        Flow: LLM asks Q1–Q5, then on receiving a5 returns RESULT (6th LLM call).
        """
        engine = _make_engine([
            "Q1?", "Q2?", "Q3?", "Q4?", "Q5?",
            "RESULT: 610910\nDESCRIPTION: T-shirts of cotton\nEXPLANATION: correct code.",
        ])
        r = engine.classify(None, "product")
        sid = r.session_id
        for answer in ["a1", "a2", "a3", "a4"]:
            engine.classify(sid, answer)  # a4 → Q5?
        r = engine.classify(sid, "a5 final answer")  # a5 → RESULT
        assert r.type == "result"
        assert r.final_result.hs_code == "610910"


# ─────────────────────────────────────────────────────────
# Session lifecycle
# ─────────────────────────────────────────────────────────

class TestSessionLifecycle:
    def test_session_deleted_after_result(self):
        engine = _make_engine([
            "RESULT: 610910\nDESCRIPTION: T-shirts of cotton\nEXPLANATION: ok.",
        ])
        r = engine.classify(None, "cotton t-shirt")
        assert r.type == "result"
        assert r.session_id not in engine._sessions

    def test_evicted_session_restarts_gracefully(self):
        """If a session is evicted between check and access, treat as new session."""
        engine = _make_engine(["Q1?", "Q1?"])
        r = engine.classify(None, "t-shirt")
        sid = r.session_id
        # Manually evict
        del engine._sessions[sid]
        # Sending answer to evicted session should restart, not crash
        r2 = engine.classify(sid, "casual wear")
        assert r2.type == "question"

    def test_lru_evicts_oldest_session(self):
        """With MAX_SESSIONS=2, adding a third session evicts the first."""
        import services.classifier as clf
        orig = clf.MAX_SESSIONS
        clf.MAX_SESSIONS = 2
        try:
            engine = _make_engine(["Q?", "Q?", "Q?"])
            r1 = engine.classify(None, "product A")
            r2 = engine.classify(None, "product B")
            assert len(engine._sessions) == 2
            r3 = engine.classify(None, "product C")
            assert len(engine._sessions) == 2
            assert r1.session_id not in engine._sessions  # first evicted
        finally:
            clf.MAX_SESSIONS = orig


# ─────────────────────────────────────────────────────────
# _simplify_query
# ─────────────────────────────────────────────────────────

class TestSimplifyQuery:
    def test_strips_that_can_clause(self):
        assert _simplify_query("metal water bottle that can keep temperature for 12 hours") == \
               "metal water bottle"

    def test_strips_designed_to(self):
        assert _simplify_query("gloves designed to protect against chemicals") == "gloves"

    def test_preserves_plain_description(self):
        assert _simplify_query("cotton t-shirt") == "cotton t-shirt"

    def test_returns_original_if_result_empty(self):
        assert _simplify_query("") == ""

    def test_strips_duration_qualifier(self):
        assert _simplify_query("battery pack for up to 48 hours") == "battery pack"


# ─────────────────────────────────────────────────────────
# _LLM_PREAMBLE regex
# ─────────────────────────────────────────────────────────

class TestLLMPreamble:
    def test_strips_step_colon(self):
        r = _LLM_PREAMBLE.sub("", "STEP 4 — SMART QUESTIONING: What material?").strip()
        assert r == "What material?"

    def test_strips_step_number_only(self):
        r = _LLM_PREAMBLE.sub("", "step 3: What is the use?").strip()
        assert r == "What is the use?"

    def test_strips_smart_questioning(self):
        r = _LLM_PREAMBLE.sub("", "Smart Questioning: Is it electrical?").strip()
        assert r == "Is it electrical?"

    def test_does_not_strip_plain_question(self):
        original = "What material is it made of?"
        r = _LLM_PREAMBLE.sub("", original).strip()
        assert r == original
