from __future__ import annotations

import asyncio
import functools
import os
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

torch.set_num_threads(1)          # one thread per inference; concurrency comes from the pool

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

_MAX_CONCURRENT = int(os.getenv("EMBED_CONCURRENCY", "4"))
_sem = asyncio.Semaphore(_MAX_CONCURRENT)


@functools.lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def embed_sync(text: str) -> np.ndarray:
    vec = _model().encode(text, normalize_embeddings=True)
    return vec.astype(np.float32)


async def embed(text: str) -> np.ndarray:
    from gateway import metrics as m

    t0 = time.perf_counter()
    async with _sem:
        waited = time.perf_counter() - t0
        m.EMBED_QUEUE_WAIT.observe(waited)
        t1 = time.perf_counter()
        vec = await asyncio.to_thread(embed_sync, text)
        m.EMBED_LATENCY.observe(time.perf_counter() - t1)
        return vec