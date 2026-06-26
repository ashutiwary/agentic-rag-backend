# RAG Chatbot Backend

Backend for a Retrieval Augmented Generation (RAG) chatbot. You give it your own documents (PDF, Word, CSV, plain text), and it answers questions about them in natural language. The system finds the most relevant pieces of your documents and feeds them to a language model so the answers stay grounded in your content.

This README is built up in stages as the code is committed. It currently covers the full core engine, from reading documents through to the RAG answer pipeline. The web API and the runnable entry points (CLI and server) are added in later stages.

## Table of contents

1. Overview
2. Tech stack
3. Prerequisites
4. Installation
5. Configuration
6. Project structure
7. Modules
8. Sample document

## 1. Overview

How a question becomes an answer:

```
Documents (PDF, DOCX, CSV, TXT)
   -> loader.py    reads raw text from each file
   -> chunker.py   splits text into overlapping chunks
   -> embedder.py  turns each chunk into a vector
   -> retriever.py stores the vectors in ChromaDB

At question time:
   question -> embedder.py -> retriever.py finds the closest chunks
            -> rag.py builds a prompt with those chunks as context
            -> llm.py streams the answer from the model
```

The `indexer.py` module ties the ingestion side together (load, chunk, embed, store) and keeps a small registry so unchanged files are skipped on re-runs.

## 2. Tech stack

- **Python 3.10 or newer.** Uses modern type hints like `list[dict]` and `str | None`.
- **ChromaDB.** Local on-disk vector database. Runs inside the Python process, no separate server.
- **sentence-transformers.** Produces the embeddings. Default model `all-MiniLM-L6-v2`, runs on CPU.
- **Groq.** Hosts the language model. Default `qwen/qwen3-32b`. Needs a free Groq API key.
- **PyMuPDF, python-docx, pandas.** Read PDF, Word, and CSV files.
- **python-dotenv.** Loads the API key from a `.env` file.

## 3. Prerequisites

- Python 3.10 or newer on your PATH. Check with `python --version`.
- pip (comes with Python).
- Around 1 GB of free disk space, since PyTorch and the transformer libraries are large.
- A Groq API key. Free to start from the Groq console.
- An internet connection on first run so the embedding model can download and cache. It runs offline after that.

## 4. Installation

From the `backend` folder:

```bash
# 1. create and activate a virtual environment
python -m venv venv
venv\Scripts\Activate.ps1        # Windows (PowerShell)
# source venv/bin/activate       # macOS / Linux

# 2. install dependencies (pulls in PyTorch, takes a few minutes)
pip install -r requirements.txt

# 3. create your .env file (see next section)
```

## 5. Configuration

The API key is read from a `.env` file in the `backend` folder. It is not committed (it is in `.gitignore`). Create it with one line:

```
GROQ_API_KEY=your_groq_api_key_here
```

Everything else lives in `app/config.py`, which loads the `.env` and creates the `data`, `data/chroma_db`, and `documents` folders on first import. Values worth tuning:

- `CHUNK_SIZE` (`500`) and `OVERLAP` (`100`): chunk length and how much neighbouring chunks share. Overlap must be smaller than chunk size.
- `EMBED_MODEL` (`all-MiniLM-L6-v2`): the embedding model. Changing it after indexing means you must reindex.
- `MODEL` (`qwen/qwen3-32b`), `TEMPERATURE` (`0.3`), `MAX_TOKENS` (`1024`): language model settings.
- `SIMILARITY_THRESHOLD` (`0.4`) and `TOP_K` (`5`): how strict retrieval is and how many chunks it pulls. If the bot too often says it has no information, lower the threshold or raise top_k.

## 6. Project structure

```
backend/
  requirements.txt   Pinned list of all Python dependencies.
  .env               Your secret Groq API key (you create this; not committed).
  .gitignore         Tells git which files to ignore.

  app/               The core engine. No web code lives here.
    config.py        All settings and paths in one place.
    loader.py        Reads raw text out of PDF, DOCX, CSV, and TXT files.
    chunker.py       Splits text into overlapping chunks.
    embedder.py      Turns text into embedding vectors.
    retriever.py     Talks to ChromaDB: stores, searches, reports stats.
    indexer.py       Orchestrates load, chunk, embed, store, with a registry.
    llm.py           Talks to the Groq model and streams replies.
    rag.py           The RAG pipeline and the RAGSession conversation class.

  documents/         Source documents. knowledge.txt ships as a sample.
  data/              Generated at runtime. Not committed.
```

The `app` folder is the engine and knows nothing about HTTP, so the same logic can be used from a script, a CLI, or a web server added later.

## 7. Modules

**config.py.** Single source of truth for settings and paths. Loads `.env` and bootstraps the required directories so a fresh checkout works without manual setup.

**loader.py.** Reads four file types and returns plain text. `load_document` picks the right reader from the file extension and raises a clear error for unsupported types. Supported extensions are exported as `SUPPORTED_FORMATS`.

**chunker.py.** `chunk_text` slices text into target-size pieces with overlap, backing up to the last space so words are not cut, with a guard that always makes forward progress.

**embedder.py.** Loads the sentence-transformer model once and reuses it. `embed` converts a list of texts into vectors; `embed_one` converts a single query into the shape ChromaDB expects.

**retriever.py.** Owns the persistent ChromaDB collection (cosine similarity). `search` embeds the query, asks for the top matches, converts distance into a 0 to 1 similarity score, drops weak matches below the threshold, and returns the rest best first. `get_stats` reports the chunk count.

**indexer.py.** The ingestion conductor. `index_document` runs load, chunk, embed, store for one file, deleting any old chunks for that file first so re-indexing replaces stale content. `index_folder` does this for every supported file, skipping unchanged ones using a JSON registry. `delete_document` removes a file's chunks.

**llm.py.** Thin layer over Groq. Holds the system prompt, builds a fresh history, and exposes `stream_chat` which yields the answer in pieces as they arrive.

**rag.py.** The RAG brain. `build_rag_prompt` wraps the retrieved chunks as context and tells the model to answer only from them, or to say it has no such information when nothing relevant is found. `RAGSession` represents one conversation, keeping its own history and listing the source files behind each answer.

## 8. Sample document

`documents/knowledge.txt` ships with the repo so there is something to index and query while testing. Drop your own PDF, DOCX, CSV, or TXT files into this same folder.

---

If anything here does not match the code, trust the code and update this file. It is meant to stay in step with the project as it grows.
