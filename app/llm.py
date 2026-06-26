from __future__ import annotations
from groq import Groq
from app.config import GROQ_API_KEY, MODEL, TEMPERATURE, MAX_TOKENS

# Single Groq client instance per process
_client: Groq = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "Answer questions using ONLY the context provided to you. "
    "If the context does not contain sufficient information to answer "
    "the question, respond with exactly: "
    "'I do not have that information in my documents.' "
    "Be concise and precise. Always state which document your answer "
    "is derived from."
)


def build_history() -> list[dict]:
    """Return a fresh conversation history seeded with the system prompt."""
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def stream_chat(messages: list[dict]):
    """
    Stream a chat completion from the LLM.

    Args:
        messages: Full conversation history including system prompt.

    Yields:
        Text chunks (str) as they arrive from the API.
    """
    stream = _client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=True,
    )

    for chunk in stream:
        text = chunk.choices[0].delta.content
        if text:
            yield text