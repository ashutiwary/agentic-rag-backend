from __future__ import annotations
import json
import logging
from groq         import Groq
from app.config   import GROQ_API_KEY, MODEL, MAX_TOKENS
from app.llm      import build_history
from app.tools    import TOOL_SCHEMAS, TOOL_FUNCTIONS

logger = logging.getLogger(__name__)

_client: Groq = Groq(api_key=GROQ_API_KEY)

AGENT_SYSTEM_PROMPT = (
    "You are an intelligent AI assistant with access to tools. "
    "Use tools when necessary to provide accurate, up-to-date answers. "
    "For questions about internal documents, use search_documents. "
    "For current events or general web information, use search_web. "
    "For mathematical calculations, use calculate. "
    "For date and time questions, use get_current_date. "
    "If you can answer directly without tools, do so. "
    "Always be concise and cite your sources."
)

MAX_TOOL_ITERATIONS = 5   # prevent infinite loops


# ---------------------------------------------------------------------------
# Core agent loop
# ---------------------------------------------------------------------------

def run_tool_call(tool_name: str, tool_args: dict) -> str:
    """
    Look up the tool function and execute it.

    Args:
        tool_name: Name matching a key in TOOL_FUNCTIONS.
        tool_args: Arguments parsed from the LLM tool call.

    Returns:
        String result to feed back to the LLM.
    """
    if tool_name not in TOOL_FUNCTIONS:
        return f"Error: unknown tool '{tool_name}'"

    func = TOOL_FUNCTIONS[tool_name]

    try:
        result = func(**tool_args)
        logger.info("Tool '%s' executed. args=%s", tool_name, tool_args)
        return result
    except Exception as exc:
        logger.error("Tool '%s' failed: %s", tool_name, exc)
        return f"Tool execution error: {exc}"


def agent_chat(
    messages: list[dict],
    stream_final: bool = True,
):
    """
    Run the agentic tool loop.

    The loop:
        1. Send messages + tool schemas to LLM.
        2. If LLM returns tool_calls — execute each tool,
           append results to history, repeat.
        3. If LLM returns a text response — stream it to caller.

    Args:
        messages:     Full conversation history.
        stream_final: If True, yields text tokens for streaming.
                      If False, returns the complete response string.

    Yields (stream_final=True):
        Text tokens as they arrive.

    Returns (stream_final=False):
        Complete response string.
    """
    iteration    = 0
    tool_history = list(messages)   # work on a copy

    while iteration < MAX_TOOL_ITERATIONS:
        iteration += 1
        logger.info("Agent iteration %d", iteration)

        # ── Call LLM with tool schemas ────────────────────────────────────
        response = _client.chat.completions.create(
            model=MODEL,
            messages=tool_history,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",     # LLM decides whether to use a tool
            max_tokens=MAX_TOKENS,
            temperature=0.3,
            parallel_tool_calls=False,
        )

        message      = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # ── LLM chose to call one or more tools ───────────────────────────
        if finish_reason == "tool_calls" and message.tool_calls:

            # Append the assistant's tool_call message to history
            tool_history.append({
                "role"      : "assistant",
                "content"   : message.content or "",
                "tool_calls": [
                    {
                        "id"      : tc.id,
                        "type"    : "function",
                        "function": {
                            "name"     : tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            })

            # Execute each tool and append results
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                logger.info(
                    "Calling tool: %s | args: %s", tool_name, tool_args
                )

                result = run_tool_call(tool_name, tool_args)

                # Append tool result to history
                tool_history.append({
                    "role"       : "tool",
                    "tool_call_id": tool_call.id,
                    "content"    : result,
                })

            # Loop continues — LLM will now read the tool results
            continue

        # ── LLM produced a final text response ───────────────────────────
        if finish_reason in ("stop", "length"):
            # The answer is already in `message.content`; re-calling the
            # model just to stream it wastes a request and can produce a
            # different answer, so reuse what we already have.
            reply = message.content or ""
            messages.append({"role": "assistant", "content": reply})

            if stream_final:
                yield reply
                return
            return reply

        break   # unexpected finish_reason — exit loop

    # Max iterations reached
    fallback = "I was unable to complete the task within the allowed steps."
    logger.warning("Agent reached max iterations (%d)", MAX_TOOL_ITERATIONS)
    messages.append({"role": "assistant", "content": fallback})
    if stream_final:
        yield fallback
    else:
        return fallback


# ---------------------------------------------------------------------------
# Session wrapper — mirrors RAGSession interface
# ---------------------------------------------------------------------------

class AgentSession:
    """
    Manages a single user's agent conversation.
    Drop-in replacement for RAGSession in the API layer.
    """

    def __init__(self) -> None:
        self._history: list[dict] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT}
        ]

    def chat(self, user_message: str):
        """
        Add user message and run the agent loop.

        Returns:
            Generator yielding text tokens.
        """
        self._history.append({
            "role"   : "user",
            "content": user_message,
        })

        return agent_chat(self._history, stream_final=True)

    def clear(self) -> None:
        """Reset conversation history."""
        self._history = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT}
        ]

    @property
    def history(self) -> list[dict]:
        return list(self._history)