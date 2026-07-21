from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

import numpy as np
import redis.asyncio as redis
from redis.commands.search.field import NumericField, TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from gateway.embeddings import EMBED_DIM, embed
from gateway.types import ChatRequest, ChatResponse, Usage

INDEX_NAME = "idx:cache"
KEY_PREFIX = "cache:entry:"


@dataclass
class CacheConfig:
    similarity_threshold: float = 0.95
    ttl_seconds: int = 3600
    enabled: bool = True


@dataclass
class CacheHit:
    response: ChatResponse
    similarity: float
    original_prompt: str


def partition_key(req: ChatRequest) -> str:
    """Exact-match dimensions. Semantic search happens only within a partition."""
    material = json.dumps({
        "system": req.system_prompt(),
        "model": req.model,
        "max_tokens": req.max_tokens,
    }, sort_keys=True)
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def exact_key(req: ChatRequest) -> str:
    material = partition_key(req) + "|" + req.last_user_message()
    return "cache:exact:" + hashlib.sha256(material.encode()).hexdigest()[:32]


class SemanticCache:
    def __init__(self, client: redis.Redis, cfg: CacheConfig | None = None):
        self.client = client
        self.cfg = cfg or CacheConfig()
        self._index_ready = False
        self._vec_cache: dict[str, np.ndarray] = {}

    async def ensure_index(self) -> None:
        if self._index_ready:
            return
        try:
            await self.client.ft(INDEX_NAME).info()
        except Exception:
            schema = (
                TagField("$.partition", as_name="partition"),
                NumericField("$.created", as_name="created"),
                VectorField(
                    "$.embedding",
                    "FLAT",
                    {"TYPE": "FLOAT32", "DIM": EMBED_DIM, "DISTANCE_METRIC": "COSINE"},
                    as_name="embedding",
                ),
            )
            await self.client.ft(INDEX_NAME).create_index(
                schema,
                definition=IndexDefinition(prefix=[KEY_PREFIX], index_type=IndexType.JSON),
            )
        self._index_ready = True

    def cacheable(self, req: ChatRequest) -> bool:
        if not self.cfg.enabled:
            return False
        if req.temperature > 0:
            return False       # caller asked for nondeterminism; honor it
        return True

    async def lookup(self, req: ChatRequest) -> CacheHit | None:
        if not self.cacheable(req):
            return None

        from gateway import metrics as m

        # Tier 1: exact match. One Redis GET, no embedding.
        raw = await self.client.get(exact_key(req))
        if raw:
            stored = json.loads(raw)
            m.CACHE_TIER.labels(tier="exact").inc()
            return CacheHit(
                response=ChatResponse(
                    content=stored["content"],
                    model=stored["model"],
                    usage=Usage(
                        prompt_tokens=stored["prompt_tokens"],
                        completion_tokens=stored["completion_tokens"],
                    ),
                    provider=stored["provider"] + " (cached)",
                ),
                similarity=1.0,
                original_prompt=req.last_user_message(),
            )

        # Tier 2: semantic match. Embedding + vector search.
        await self.ensure_index()
        vec = await embed(req.last_user_message())
        self._vec_cache[exact_key(req)] = vec       # let store() reuse it on a miss
        part = partition_key(req)

        q = (
            Query(f"(@partition:{{{part}}})=>[KNN 1 @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("score", "$.response", "$.prompt")
            .dialect(2)
        )
        try:
            res = await self.client.ft(INDEX_NAME).search(
                q, query_params={"vec": vec.tobytes()}
            )
        except Exception as e:
            if "no such index" in str(e).lower():
                self._index_ready = False       # index vanished — rebuild next call
                await self.ensure_index()
            return None

        if not res.docs:
            return None

        doc = res.docs[0]
        # RediSearch returns COSINE *distance*; similarity = 1 - distance
        similarity = 1.0 - float(doc.score)
        if similarity < self.cfg.similarity_threshold:
            return None

        self._vec_cache.pop(exact_key(req), None)    # hit -> store() won't run, don't leak it

        stored = json.loads(getattr(doc, "$.response"))
        resp = ChatResponse(
            content=stored["content"],
            model=stored["model"],
            usage=Usage(
                prompt_tokens=stored["prompt_tokens"],
                completion_tokens=stored["completion_tokens"],
            ),
            provider=stored["provider"] + " (cached)",
        )
        m.CACHE_TIER.labels(tier="semantic").inc()
        return CacheHit(
            response=resp,
            similarity=similarity,
            original_prompt=getattr(doc, "$.prompt"),
        )

    async def store(self, req: ChatRequest, resp: ChatResponse,
                     vec: np.ndarray | None = None) -> None:
        if not self.cacheable(req):
            return
        await self.ensure_index()

        prompt = req.last_user_message()
        if vec is None:
            vec = self._vec_cache.pop(exact_key(req), None)
        if vec is None:
            vec = await embed(prompt)
        key = KEY_PREFIX + hashlib.sha256(
            (partition_key(req) + prompt).encode()
        ).hexdigest()[:32]

        payload = json.dumps({
            "content": resp.content,
            "model": resp.model,
            "provider": resp.provider,
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        })
        await self.client.set(exact_key(req), payload, ex=self.cfg.ttl_seconds)

        doc = {
            "partition": partition_key(req),
            "prompt": prompt,
            "created": time.time(),
            "embedding": vec.tolist(),
            "response": payload,
        }
        await self.client.json().set(key, "$", doc)
        await self.client.expire(key, self.cfg.ttl_seconds)

    async def stats(self) -> dict:
        await self.ensure_index()
        info = await self.client.ft(INDEX_NAME).info()
        return {
            "entries": int(info["num_docs"]),
            "threshold": self.cfg.similarity_threshold,
            "ttl_seconds": self.cfg.ttl_seconds,
            "enabled": self.cfg.enabled,
        }