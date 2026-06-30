from __future__ import annotations
import os
import re
import logging
import xml.etree.ElementTree as ET

import requests

from app.config import DATA_DIR
from app.loader import load_document
from app.connectors.base import Connector, RemoteDoc

logger = logging.getLogger(__name__)

_ARXIV_API = "http://export.arxiv.org/api/query"
_ATOM      = "{http://www.w3.org/2005/Atom}"


class ArxivConnector(Connector):
    """
    Fetch papers from the public arXiv API. No API key required.

    Versioning uses arXiv's own "updated" timestamp, so a revised paper
    (v1 -> v2) is detected and re-indexed while unchanged papers are skipped.
    Each paper's PDF is downloaded lazily, parsed with the normal PDF loader,
    and the temporary file is removed afterwards.

    sources.json entry:
        { "type": "arxiv", "query": "cat:cs.AI", "max_results": 5 }

    Query syntax: https://info.arxiv.org/help/api/user-manual.html
    """
    kind = "arxiv"

    def __init__(self, query: str, max_results: int = 5):
        self.query       = query
        self.max_results = max_results

    @classmethod
    def from_spec(cls, spec: dict) -> "ArxivConnector":
        return cls(
            query=spec.get("query", "cat:cs.AI"),
            max_results=int(spec.get("max_results", 5)),
        )

    def list_documents(self) -> list[RemoteDoc]:
        params = {
            "search_query": self.query,
            "start"       : 0,
            "max_results" : self.max_results,
            "sortBy"      : "submittedDate",
            "sortOrder"   : "descending",
        }
        resp = requests.get(_ARXIV_API, params=params, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)

        docs: list[RemoteDoc] = []
        for entry in root.findall(f"{_ATOM}entry"):
            abs_url = (entry.findtext(f"{_ATOM}id") or "").strip()
            title   = " ".join((entry.findtext(f"{_ATOM}title") or "").split())
            updated = (entry.findtext(f"{_ATOM}updated") or "").strip()

            pdf_url = None
            for link in entry.findall(f"{_ATOM}link"):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href")
                    break
            if not (abs_url and pdf_url):
                continue

            arxiv_id = abs_url.rsplit("/abs/", 1)[-1]
            docs.append(RemoteDoc(
                source_id=f"arxiv:{arxiv_id}",
                version=updated,
                name=title or arxiv_id,
                fetch=self._make_fetch(pdf_url, arxiv_id),
            ))

        logger.info("arXiv listed %d paper(s) for query %r", len(docs), self.query)
        return docs

    def _make_fetch(self, pdf_url: str, arxiv_id: str):
        """Build the lazy fetch closure that downloads and parses the PDF."""
        def fetch() -> str:
            resp = requests.get(pdf_url, timeout=60)
            resp.raise_for_status()

            safe = re.sub(r"[^A-Za-z0-9.]+", "_", arxiv_id)
            tmp  = os.path.join(DATA_DIR, f"_arxiv_{safe}.pdf")
            with open(tmp, "wb") as f:
                f.write(resp.content)
            try:
                return load_document(tmp)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

        return fetch
