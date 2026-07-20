from __future__ import annotations

import asyncio
import functools

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384


@functools.lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def embed_sync(text: str) -> np.ndarray:
    vec = _model().encode(text, normalize_embeddings=True)
    return vec.astype(np.float32)


async def embed(text: str) -> np.ndarray:
    """Run the model in a thread so it never blocks the event loop."""
    return await asyncio.to_thread(embed_sync, text)