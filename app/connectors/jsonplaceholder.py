from __future__ import annotations
import hashlib
import logging

import requests

from app.connectors.base import Connector, RemoteDoc

logger = logging.getLogger(__name__)

_URL = "https://jsonplaceholder.typicode.com/posts"


class JSONPlaceholderConnector(Connector):
    """
    Pull fake "posts" from JSONPlaceholder. A smoke-test source: it proves
    the generic loop, the registry, and the skip logic work end to end
    without any credentials or file loaders.

    The API exposes no version field, so this connector hashes the content
    to produce a version. That is the universal fallback for any source that
    cannot tell you when something changed.

    sources.json entry:
        { "type": "jsonplaceholder", "limit": 10 }
    """
    kind = "jsonplaceholder"

    def __init__(self, limit: int = 20):
        self.limit = limit

    @classmethod
    def from_spec(cls, spec: dict) -> "JSONPlaceholderConnector":
        return cls(limit=int(spec.get("limit", 20)))

    def list_documents(self) -> list[RemoteDoc]:
        resp = requests.get(_URL, timeout=30)
        resp.raise_for_status()
        posts = resp.json()[: self.limit]

        docs: list[RemoteDoc] = []
        for post in posts:
            pid  = post.get("id")
            text = f"{post.get('title', '')}\n\n{post.get('body', '')}".strip()
            version = hashlib.sha256(text.encode("utf-8")).hexdigest()

            docs.append(RemoteDoc(
                source_id=f"jsonplaceholder:post:{pid}",
                version=version,
                name=f"post-{pid}",
                fetch=(lambda t=text: t),   # text is already in hand
            ))

        logger.info("JSONPlaceholder listed %d post(s)", len(docs))
        return docs
