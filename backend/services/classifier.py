"""
Classification Engine (LLM-Powered).

Purpose: Core reasoning engine that uses Google Gemini for intelligent
         conversational classification. Combines FAISS semantic search
         for candidate retrieval with LLM reasoning for question generation
         and classification decisions.

Inputs:  User messages, candidate lists from vector search.
Outputs: Clarifying questions or final classification results (via LLM).

Behavioral Rules (from gemini.md):
    - Never invent HS codes — only use codes from the dataset.
    - Never guess — ask clarifying questions when uncertain.
    - Explain reasoning behind the final classification.
    - Respect the HS hierarchy (chapter → heading → subheading).
"""

import re
import uuid

from models.schemas import (
    Candidate,
    ChatMessage,
    ClassifyResponse,
    FinalResult,
    SessionState,
)
from services.embedding import EmbeddingService
from services.hs_knowledge import HSKnowledgeBase
from services.llm_service import GeminiService
from services.vector_search import VectorSearchService
from utils.logger import get_logger

logger = get_logger("classifier")

# Search parameters
TOP_K_CANDIDATES = 10
MAX_QUESTIONS = 5


class ClassificationEngine:
    """Manages conversational HS code classification sessions using Gemini LLM."""

    def __init__(
        self,
        knowledge_base: HSKnowledgeBase,
        embedding_service: EmbeddingService,
        vector_search: VectorSearchService,
        llm_service: GeminiService,
    ) -> None:
        self._kb = knowledge_base
        self._embed = embedding_service
        self._search = vector_search
        self._llm = llm_service
        self._sessions: dict[str, SessionState] = {}

    def classify(self, session_id: str | None, message: str) -> ClassifyResponse:
        """Process a user message and return the next step in classification.

        For new sessions: performs vector search, sends candidates to Gemini.
        For existing sessions: forwards user answer to Gemini with context.

        Args:
            session_id: Existing session ID or None for new conversations.
            message: User's product description or answer to a question.

        Returns:
            ClassifyResponse with either a question or final result.
        """
        message = message.strip()
        if not message:
            return self._error_response("Please describe the product you'd like to classify.")

        # New session
        if session_id is None or session_id not in self._sessions:
            return self._start_new_session(message)

        # Existing session — process the answer via LLM
        return self._process_answer(session_id, message)

    def _start_new_session(self, description: str) -> ClassifyResponse:
        """Start a new classification session with vector search + LLM."""
        session_id = str(uuid.uuid4())
        logger.info("New session %s started (description_length=%d)", session_id[:8], len(description))

        # Create session state
        session = SessionState(
            session_id=session_id,
            original_query=description,
        )
        session.messages.append(ChatMessage(role="user", content=description))

        # Perform vector search
        try:
            query_vector = self._embed.encode(description)
            candidates = self._search.search(query_vector, top_k=TOP_K_CANDIDATES)
        except Exception as e:
            logger.error("Search failed: %s", e)
            return self._fallback_response(session_id)

        if not candidates:
            return self._fallback_response(session_id)

        session.candidates = candidates
        self._sessions[session_id] = session

        # Format candidates for LLM
        candidates_text = self._format_candidates(candidates)

        # Ask Gemini to either classify or ask a question
        llm_response = self._llm.generate_response(
            user_query=description,
            candidates_text=candidates_text,
            conversation_history=[],
        )

        # Store LLM response in conversation history
        session.llm_history = [
            {"role": "user", "parts": [
                f'The user wants to classify this product: "{description}"\n\n'
                f'Here are the top candidate HS codes from semantic search:\n\n'
                f'{candidates_text}\n\n'
                f'Based on these candidates, either classify the product directly '
                f'if you\'re confident, or ask ONE short clarifying question to '
                f'narrow it down.'
            ]},
            {"role": "model", "parts": [llm_response]},
        ]

        return self._parse_llm_response(session, llm_response)

    def _process_answer(self, session_id: str, answer: str) -> ClassifyResponse:
        """Process a user's answer by forwarding to Gemini with context."""
        session = self._sessions[session_id]
        session.messages.append(ChatMessage(role="user", content=answer))
        session.questions_asked += 1

        # Check question limit
        if session.questions_asked >= MAX_QUESTIONS:
            logger.info("Max questions reached for session %s", session_id[:8])
            # Force Gemini to decide
            top = session.candidates[0] if session.candidates else None
            if top:
                return self._force_result(session, top)

        # Re-embed the combined query (original + answer) for better candidates
        combined_query = f"{session.original_query} {answer}"
        try:
            query_vector = self._embed.encode(combined_query)
            new_candidates = self._search.search(query_vector, top_k=TOP_K_CANDIDATES)
            if new_candidates:
                session.candidates = new_candidates
        except Exception as e:
            logger.warning("Re-search failed, keeping existing candidates: %s", e)

        candidates_text = self._format_candidates(session.candidates)

        # Get conversation history
        history = getattr(session, 'llm_history', [])

        # Ask Gemini for next step
        llm_response = self._llm.generate_response(
            user_query=answer,
            candidates_text=candidates_text,
            conversation_history=history,
        )

        # Update conversation history
        if not hasattr(session, 'llm_history'):
            session.llm_history = []
        session.llm_history.append({"role": "user", "parts": [
            f'User\'s answer: "{answer}"\n\nRemaining candidates:\n{candidates_text}\n\n'
            f'Based on this answer, either classify the product or ask another short question.'
        ]})
        session.llm_history.append({"role": "model", "parts": [llm_response]})

        return self._parse_llm_response(session, llm_response)

    def _parse_llm_response(self, session: SessionState, llm_text: str) -> ClassifyResponse:
        """Parse the LLM response to determine if it's a result or question.

        Checks for the RESULT: format pattern. If found, extracts the HS code
        and validates it against the dataset. Otherwise treats as a question.
        """
        # Check if LLM returned a classification result
        result_match = re.search(
            r'RESULT:\s*(\d{4,6})',
            llm_text,
            re.IGNORECASE,
        )

        if result_match:
            hs_code = result_match.group(1)
            return self._build_result(session, hs_code, llm_text)

        # It's a question — clean up the response
        question = llm_text.strip()
        # Remove any "QUESTION:" prefix if present
        question = re.sub(r'^(QUESTION|Q)\s*:\s*', '', question, flags=re.IGNORECASE).strip()

        session.messages.append(ChatMessage(role="assistant", content=question))

        return ClassifyResponse(
            session_id=session.session_id,
            type="question",
            message=question,
            candidates=[],
            final_result=None,
        )

    def _build_result(self, session: SessionState, hs_code: str, llm_text: str) -> ClassifyResponse:
        """Build a final classification result from the LLM's decision."""
        session.is_complete = True

        # Validate HS code exists in dataset
        entry = self._kb.get_by_code(hs_code)
        if not entry:
            # Try to find the closest matching candidate
            for c in session.candidates:
                if c.hs_code == hs_code:
                    entry_desc = c.description
                    break
            else:
                # Code not found — use LLM's description
                entry_desc = "Classification result"
        else:
            entry_desc = entry.description

        # Extract explanation from LLM response
        explanation_match = re.search(
            r'EXPLANATION:\s*(.+?)(?:\n|$)',
            llm_text,
            re.IGNORECASE | re.DOTALL,
        )
        explanation = explanation_match.group(1).strip() if explanation_match else ""

        # Build hierarchy path
        hierarchy = self._kb.get_hierarchy_path(hs_code)
        hierarchy_text = " → ".join(
            f"{e.hs_code}: {e.description}" for e in hierarchy
        ) if hierarchy else ""

        full_explanation = explanation
        if hierarchy_text:
            full_explanation += f"\n\n**Classification path:** {hierarchy_text}"

        message = (
            f"I've classified your product as **HS Code {hs_code}**.\n\n"
            f"📦 **{entry_desc}**\n\n"
            f"{full_explanation}"
        )

        session.messages.append(ChatMessage(role="assistant", content=message))

        logger.info("Session %s result: %s", session.session_id[:8], hs_code)

        return ClassifyResponse(
            session_id=session.session_id,
            type="result",
            message=message,
            candidates=[c for c in session.candidates if c.hs_code == hs_code][:1] or session.candidates[:1],
            final_result=FinalResult(
                hs_code=hs_code,
                description=entry_desc,
                explanation=full_explanation,
            ),
        )

    def _force_result(self, session: SessionState, candidate: Candidate) -> ClassifyResponse:
        """Force a result when max questions are reached."""
        hierarchy = self._kb.get_hierarchy_path(candidate.hs_code)
        hierarchy_text = " → ".join(
            f"{e.hs_code}: {e.description}" for e in hierarchy
        ) if hierarchy else ""

        explanation = (
            f"Based on your description and answers, this is the best matching code."
        )
        if hierarchy_text:
            explanation += f"\n\n**Classification path:** {hierarchy_text}"

        message = (
            f"I've classified your product as **HS Code {candidate.hs_code}**.\n\n"
            f"📦 **{candidate.description}**\n\n"
            f"{explanation}"
        )

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

    def _format_candidates(self, candidates: list[Candidate]) -> str:
        """Format candidates into a readable string for the LLM."""
        lines = []
        for i, c in enumerate(candidates, 1):
            lines.append(
                f"{i}. HS {c.hs_code} — {c.description} "
                f"(similarity: {c.similarity_score:.0%}, level: {c.level})"
            )
        return "\n".join(lines)

    def _fallback_response(self, session_id: str) -> ClassifyResponse:
        """Fallback when vector search fails."""
        return ClassifyResponse(
            session_id=session_id,
            type="question",
            message=(
                "I couldn't find a close match. Could you describe the product "
                "differently? Mention what it's made of, what it's used for, "
                "or what category it might fall into."
            ),
            candidates=[],
            final_result=None,
        )

    def _error_response(self, message: str) -> ClassifyResponse:
        """Return a friendly error response."""
        return ClassifyResponse(
            session_id=str(uuid.uuid4()),
            type="question",
            message=message,
            candidates=[],
            final_result=None,
        )
