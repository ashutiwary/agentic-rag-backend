from __future__ import annotations
import os
import json
import logging

from app.config import SOURCES_FILE
from app.connectors.base import Connector, RemoteDoc
from app.connectors.arxiv import ArxivConnector
from app.connectors.jsonplaceholder import JSONPlaceholderConnector

logger = logging.getLogger(__name__)

# Registry of available connector types. To add a new source type, write a
# Connector subclass in this package and register it here under its kind.
# To remove one, delete its module and drop the line below.
CONNECTORS: dict[str, type[Connector]] = {
    ArxivConnector.kind          : ArxivConnector,
    JSONPlaceholderConnector.kind: JSONPlaceholderConnector,
}


def build_connector(spec: dict) -> Connector:
    """Instantiate the connector named by spec['type']."""
    kind = spec.get("type")
    cls  = CONNECTORS.get(kind)
    if cls is None:
        raise ValueError(
            f"Unknown source type '{kind}'. Available: {sorted(CONNECTORS)}"
        )
    return cls.from_spec(spec)


def load_sources() -> list[dict]:
    """
    Read the active source list from sources.json. A missing file means no
    external sources are configured, which is not an error.
    """
    if not os.path.exists(SOURCES_FILE):
        return []
    with open(SOURCES_FILE, "r", encoding="utf-8") as f:
        sources = json.load(f)
    if not isinstance(sources, list):
        raise ValueError("sources.json must contain a JSON list of sources")
    return sources


__all__ = ["Connector", "RemoteDoc", "CONNECTORS", "build_connector", "load_sources"]
