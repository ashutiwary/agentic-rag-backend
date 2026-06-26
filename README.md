# RAG Chatbot Backend

Backend for a Retrieval Augmented Generation (RAG) chatbot. You give it your own documents (PDF, Word, CSV, plain text), and later you can ask questions about them in natural language. The system finds the most relevant pieces of your documents and feeds them to a language model so the answers stay grounded in your content.

This README is built up in stages as the code is committed. Right now it covers the project setup, configuration, and the document loading and chunking layer. Later sections (embeddings, indexing, the LLM, the API, and how to run the server) will be added as those parts land.

## Table of contents

1. Overview
2. Tech stack
3. Prerequisites
4. Installation
5. Configuration and the .env file
6. Project structure
7. Modules so far
8. Sample document

## 1. Overview

The core job of this backend is to answer questions about a private set of documents. You put files into a folder, the system reads them, breaks them into small pieces, and (in a later stage) converts each piece into a numeric vector stored in a local database. When a question comes in, the closest matching pieces are handed to the language model as context.

The work so far covers the first part of that flow:

- Reading raw text out of different file types (`loader.py`).
- Splitting that text into overlapping chunks ready for embedding (`chunker.py`).

Everything is configured from one place (`config.py`) so paths and settings are not scattered across the code.

## 2. Tech stack

- **Python 3.10 or newer.** The code uses modern type hints like `list[dict]` and `str | None`.
- **PyMuPDF, python-docx, pandas.** Used to read PDF, Word, and CSV files.
- **python-dotenv.** Loads secrets from a `.env` file.
- **ChromaDB, sentence-transformers, FastAPI, Groq.** Listed in `requirements.txt` and used by parts of the project that arrive in later stages.

## 3. Prerequisites

- Python 3.10 or newer on your PATH. Check with `python --version`.
- pip (comes with Python).
- Around 1 GB of free disk space, since the dependencies (PyTorch, transformers) are large.
- A Groq API key for the later chat stages. Not needed yet for loading and chunking.

## 4. Installation

These commands assume you are in the `backend` folder.

**Step 1. Create a virtual environment.**

On Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Your prompt should now show `(venv)` at the start.

**Step 2. Install the dependencies.**

```bash
pip install -r requirements.txt
```

This pulls in PyTorch and the transformer libraries, so it takes a few minutes.

**Step 3. Create your `.env` file.** See the next section.

## 5. Configuration and the .env file

The project reads secrets from a file named `.env` in the `backend` folder. It is not committed to version control (it is listed in `.gitignore`) because it holds your private key.

Create a file called `.env` with this line:

```
GROQ_API_KEY=your_groq_api_key_here
```

Replace the value with your real key. That is the only secret the project needs. Everything else is set in code in `app/config.py`.

The `.env` is loaded at the top of `app/config.py`, which finds it relative to the project folder so it works no matter which directory you launch Python from.

## 6. Project structure

Only the files committed so far are listed here. The tree grows as later stages are added.

```
backend/
  requirements.txt   Pinned list of all Python dependencies.
  .env               Your secret Groq API key (you create this; not committed).
  .gitignore         Tells git which files to ignore.

  app/               The core engine. No web code lives here.
    config.py        All settings and paths in one place.
    loader.py        Reads raw text out of PDF, DOCX, CSV, and TXT files.
    chunker.py       Splits text into overlapping chunks.

  documents/         Source documents. knowledge.txt ships as a sample.
```

The `app` folder is the engine and knows nothing about HTTP. Keeping it separate means the same logic can be used from the command line, a web server, or your own scripts later on.

## 7. Modules so far

### app/config.py

The single source of truth for configuration. It loads the `.env` file, then defines the model name, temperature, token limits, all the folder paths, the embedding model, the chunk sizes, and the retrieval settings. At the bottom it creates the required `data` and `documents` directories if they do not already exist, so a fresh checkout works without manual setup.

The main values you can tune here:

- `CHUNK_SIZE` (`500`): target characters per chunk. Larger chunks carry more context but match less precisely.
- `OVERLAP` (`100`): characters shared between neighbouring chunks. Must be smaller than the chunk size.
- `MODEL`, `TEMPERATURE`, `MAX_TOKENS`: language model settings, used in later stages.
- `EMBED_MODEL`, `SIMILARITY_THRESHOLD`, `TOP_K`: embedding and retrieval settings, used in later stages.

### app/loader.py

Knows how to read four file types. Each loader function takes a file path and returns plain text. PDFs are read with PyMuPDF, Word files with python-docx, CSVs are flattened row by row into readable lines with pandas, and text files are read directly.

The single public entry point is `load_document`, which looks at the file extension, picks the right loader, and raises a clear error for unsupported types or missing files. The set of supported extensions is exported as `SUPPORTED_FORMATS`.

### app/chunker.py

Contains one function, `chunk_text`. It slices text into pieces of a target size with a configurable overlap between consecutive pieces. The overlap keeps context from being lost at the boundary between two chunks. The function avoids cutting in the middle of a word by backing up to the last space, and it has a guard that guarantees forward progress so it cannot get stuck on awkward input.

## 8. Sample document

`documents/knowledge.txt` ships with the repo as a sample file so there is something to load and chunk while testing. You can drop your own PDF, DOCX, CSV, or TXT files into this same folder.
