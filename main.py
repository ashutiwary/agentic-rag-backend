from __future__ import annotations
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def run_cli() -> None:
    """Interactive CLI for local testing without the HTTP layer."""
    from app.rag       import RAGSession
    from app.retriever import get_stats

    session = RAGSession()
    stats   = get_stats()

    print(f"RAG Chatbot | {stats['total_chunks']} chunks indexed")
    print("Commands: quit | clear | stats\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession terminated.")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye.")
            break

        if user_input.lower() == "clear":
            session.clear()
            print("Conversation history cleared.\n")
            continue

        if user_input.lower() == "stats":
            s = get_stats()
            print(
                f"Chunks: {s['total_chunks']} | "
                f"Collection: {s['collection']} | "
                f"Model: {s['embed_model']}\n"
            )
            continue

        try:
            hits, stream = session.chat(user_input)
            sources      = session.extract_sources(hits)

            if sources:
                print(f"Sources: {', '.join(sources)}")

            print("Assistant: ", end="", flush=True)
            for token in stream:
                print(token, end="", flush=True)
            print("\n")

        except Exception as exc:
            logger.error("Chat error: %s", exc)
            print(f"Error: {exc}\n")


def run_index() -> None:
    """Index all documents in the documents folder."""
    from app.indexer import index_folder

    summary = index_folder()
    print(
        f"Indexing complete. "
        f"indexed={summary['indexed']} "
        f"skipped={summary['skipped']} "
        f"failed={summary['failed']} "
        f"total_chunks={summary['total_chunks']}"
    )


def run_server() -> None:
    """Start the FastAPI server with uvicorn."""
    import uvicorn
    
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    if "--index" in sys.argv:
        run_index()
    elif "--server" in sys.argv:
        run_server()
    else:
        run_cli()