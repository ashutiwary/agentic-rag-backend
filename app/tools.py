from __future__ import annotations
import logging
import math
from datetime          import datetime
from ddgs              import DDGS
from app.retriever     import search as vector_search

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool implementations
# Plain Python functions. Each returns a string — the result that gets
# fed back to the LLM as tool output.
# ---------------------------------------------------------------------------

def search_documents(query: str) -> str:
    """
    Search internal ChromaDB vector store.
    Returns top relevant chunks as formatted text.
    """
    hits = vector_search(query)

    if not hits:
        return "No relevant information found in the documents."

    parts = []
    for i, hit in enumerate(hits, start=1):
        parts.append(
            f"[{i}] Source: {hit['source']} | Relevance: {hit['score']}\n"
            f"{hit['text']}"
        )

    return "\n\n".join(parts)


def search_web(query: str) -> str:
    """
    Search the web via DuckDuckGo.
    Returns top 3 results as formatted text.
    """
    try:
        results = DDGS().text(query, max_results=3)
    except Exception as exc:
        logger.error("Web search failed: %s", exc)
        return f"Web search failed: {exc}"

    if not results:
        return "No web results found."

    parts = []
    for i, r in enumerate(results, start=1):
        parts.append(
            f"[{i}] {r['title']}\n"
            f"URL: {r['href']}\n"
            f"{r['body']}"
        )

    return "\n\n".join(parts)


def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    Supports: +, -, *, /, **, sqrt, log, sin, cos, tan, pi, e
    """
    # Whitelist safe names only — never use eval() on raw input
    safe_names = {
        "sqrt" : math.sqrt,
        "log"  : math.log,
        "log10": math.log10,
        "sin"  : math.sin,
        "cos"  : math.cos,
        "tan"  : math.tan,
        "pi"   : math.pi,
        "e"    : math.e,
        "abs"  : abs,
        "round": round,
        "pow"  : pow,
    }

    try:
        result = eval(expression, {"__builtins__": {}}, safe_names)
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as exc:
        return f"Calculation error: {exc}"


def get_current_date() -> str:
    """Return the current date and time."""
    now = datetime.now()
    return (
        f"Current date and time: "
        f"{now.strftime('%A, %d %B %Y, %H:%M:%S')}"
    )


# ---------------------------------------------------------------------------
# Tool registry
# Maps tool name → Python function.
# Agent uses this to look up which function to call.
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS: dict[str, callable] = {
    "search_documents": search_documents,
    "search_web"      : search_web,
    "calculate"       : calculate,
    "get_current_date": get_current_date,
}


# ---------------------------------------------------------------------------
# Tool schemas
# JSON schema definitions sent to the LLM so it knows what tools exist,
# what they do, and what arguments they accept.
# This is the standard OpenAI / Groq tool calling format.
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name"       : "search_documents",
            "description": (
                "Search the internal document knowledge base for information. "
                "Use this when the user asks about topics that may be covered "
                "in the uploaded documents."
            ),
            "parameters": {
                "type"      : "object",
                "properties": {
                    "query": {
                        "type"       : "string",
                        "description": "The search query to look up in documents",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name"       : "search_web",
            "description": (
                "Search the web for current information, recent events, "
                "news, or anything not likely to be in internal documents."
            ),
            "parameters": {
                "type"      : "object",
                "properties": {
                    "query": {
                        "type"       : "string",
                        "description": "The search query to look up on the web",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name"       : "calculate",
            "description": (
                "Evaluate a mathematical expression. "
                "Use this for any arithmetic, percentages, or calculations."
            ),
            "parameters": {
                "type"      : "object",
                "properties": {
                    "expression": {
                        "type"       : "string",
                        "description": (
                            "A valid mathematical expression, "
                            "e.g. '15 / 100 * 42000' or 'sqrt(144)'"
                        ),
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name"       : "get_current_date",
            "description": (
                "Get the current date and time. "
                "Use this when the user asks what day or time it is, "
                "or for any date-related calculations."
            ),
            "parameters": {
                "type"      : "object",
                "properties": {},
                "required" : [],
            },
        },
    },
]