from __future__ import annotations
from app.retriever import search
from app.llm       import stream_chat, build_history
from typing        import Generator


def build_rag_prompt(question: str, hits: list[dict]) -> str:
    """
    Construct a prompt that injects retrieved document chunks
    as grounding context before the user question.

    Args:
        question: The original user question.
        hits:     Retrieved chunks from the vector store.

    Returns:
        A formatted prompt string ready to send to the LLM.
    """
    if not hits:
        return (
            "No relevant context was found in the documents for the question "
            "below. You MUST respond with exactly: "
            "'I do not have that information in my documents.'\n\n"
            f"QUESTION: {question}"
        )

    context_blocks = []
    for i, hit in enumerate(hits, start=1):
        context_blocks.append(
            f"[{i}] Source: {hit['source']} | Relevance: {hit['score']}\n"
            f"{hit['text']}"
        )

    context = "\n\n".join(context_blocks)

    return (
        "Use the following context extracted from documents to answer "
        "the question below. If the answer is not present in the context, "
        "respond with: 'I do not have that information in my documents.'\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}"
    )


class RAGSession:
    """
    Encapsulates a single user conversation session.

    Each session maintains its own conversation history so that
    multiple concurrent users (API sessions) remain isolated.
    """

    def __init__(self) -> None:
        self._history: list[dict] = build_history()

    def chat(
        self,
        user_message: str,
    ) -> tuple[list[dict], Generator[str, None, None]]:
        """
        Execute the full RAG pipeline for one user turn.

        Pipeline:
            1. Retrieve relevant chunks from the vector store.
            2. Build a context-injected prompt.
            3. Append the prompt to conversation history.
            4. Stream the LLM response.
            5. Append the completed response to history.

        Args:
            user_message: Raw message from the user.

        Returns:
            Tuple of (hits, token_stream) where:
                hits         — list of retrieved chunk dicts
                token_stream — generator yielding text chunks
        """
        hits       = search(user_message)
        rag_prompt = build_rag_prompt(user_message, hits)

        self._history.append({"role": "user", "content": rag_prompt})

        full_reply: list[str] = []

        def _stream() -> Generator[str, None, None]:
            for token in stream_chat(self._history):
                full_reply.append(token)
                yield token

            # Store completed reply in history after stream finishes
            self._history.append({
                "role"   : "assistant",
                "content": "".join(full_reply),
            })

        return hits, _stream()

    def clear(self) -> None:
        """Reset conversation history, preserving the system prompt."""
        self._history = build_history()

    @property
    def history(self) -> list[dict]:
        """Read-only view of current conversation history."""
        return list(self._history)

    @staticmethod
    def extract_sources(hits: list[dict]) -> list[str]:
        """Return deduplicated list of source filenames from hits."""
        seen   = set()
        result = []
        for h in hits:
            src = h["source"]
            if src not in seen:
                seen.add(src)
                result.append(src)
        return result