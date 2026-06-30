from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable


@dataclass
class RemoteDoc:
    """
    One document from an external source.

    fetch() is called lazily by the sync engine, only when the document is
    new or its version changed, so unchanged documents are never downloaded.

    Fields:
        source_id: Stable unique id, namespaced by source (e.g.
                   "arxiv:2401.01234"). Used for storage and change detection.
        version:   Token that changes when the content changes: a remote
                   timestamp, an etag, or a content hash.
        name:      Human-readable display name, shown as the answer source.
        fetch:     Zero-arg callable returning the document text.
    """
    source_id: str
    version: str
    name: str
    fetch: Callable[[], str]


class Connector(ABC):
    """
    Base class for an external document source.

    A connector only has to list its documents with an id, a version, and a
    way to fetch the text. Everything downstream (chunk, embed, store, change
    detection, deletion) is handled by the sync engine and the indexer.

    kind is a short stable label stored on every chunk and registry entry so
    sync can find and clean up documents belonging to this source type.
    """
    kind: str = "base"

    @classmethod
    @abstractmethod
    def from_spec(cls, spec: dict) -> "Connector":
        """Build a connector instance from one entry in sources.json."""
        raise NotImplementedError

    @abstractmethod
    def list_documents(self) -> list[RemoteDoc]:
        """Return the current set of documents this source exposes."""
        raise NotImplementedError
