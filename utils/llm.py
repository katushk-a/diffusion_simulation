"""
Thin LLM backend wrapper supporting:
  - OpenAI API  (default)
  - Ollama      (local, set backend="ollama")
  - Mock        (for unit tests / offline runs)

All methods are async. JSON parsing falls back to raw text if schema fails.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_fences(raw: str) -> str:
    """Remove markdown code fences and leading/trailing whitespace."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        # parts[1] is the fenced block content (may start with "json\n")
        raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def _schema_to_example(schema: Type) -> dict:
    """
    Build a field-description dict from a Pydantic model.
    Uses descriptive strings as placeholders so the model understands
    it must decide, not copy the example.
    """
    mapping = {
        "boolean": "true or false",
        "string": "<your text here>",
        "integer": "<integer>",
        "number": "<number>",
        "array": [],
        "object": {},
    }
    props = schema.model_json_schema().get("properties", {})
    example = {}
    for field, info in props.items():
        ftype = info.get("type", "string")
        if isinstance(ftype, list):          # e.g. ["string", "null"]
            ftype = next((t for t in ftype if t != "null"), "string")
        example[field] = mapping.get(ftype, f"<{ftype}>")
    return example


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMBackend(ABC):
    @abstractmethod
    async def complete(self, prompt: str, temperature: float = 0.7) -> str:
        """Return a text completion for prompt."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a list of texts."""

    async def complete_json(
        self,
        prompt: str,
        schema: Type[T],
        temperature: float = 0.3,
    ) -> T:
        """
        Complete with JSON output. Builds a concrete example from the schema
        so the model understands it should fill in values, not reproduce the schema.
        """
        example = _schema_to_example(schema)
        json_prompt = (
            prompt
            + "\n\nRespond ONLY with a single JSON object. "
            "No explanation, no markdown fences, no extra text.\n"
            f"Required fields and types:\n{json.dumps(example, indent=2)}"
        )
        raw = await self.complete(json_prompt, temperature=temperature)
        raw = _strip_fences(raw)
        return schema.model_validate_json(raw)


# ---------------------------------------------------------------------------
# OpenAI-compatible backend (works with OpenAI, MetaCentrum, etc.)
# ---------------------------------------------------------------------------

class OpenAICompatibleBackend(LLMBackend):
    """
    Generic backend for any OpenAI-compatible API.

    MetaCentrum example:
        OpenAICompatibleBackend(
            model="meta-llama/Llama-3.3-70B-Instruct",
            base_url="https://llm.metacentrum.cz/v1",
            api_key_env="METACENTRUM_TOKEN",
        )
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        embedding_model: Optional[str] = None,
        max_retries: int = 6,
        request_delay: float = 0.0,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError("Install openai: pip install openai") from e

        self.model = model
        self.embedding_model = embedding_model
        # max_retries: SDK retries on 429/5xx with exponential backoff automatically.
        # Raise from default 2 to give rate-limited APIs more time to recover.
        self.client = AsyncOpenAI(
            api_key=api_key or os.environ.get(api_key_env),
            base_url=base_url,
            max_retries=max_retries,
        )
        # Optional fixed delay between every request — useful for APIs that
        # throttle silently without returning a proper 429 (e.g. MetaCentrum).
        self._request_delay = request_delay

    async def complete(self, prompt: str, temperature: float = 0.7) -> str:
        import asyncio
        if self._request_delay > 0:
            await asyncio.sleep(self._request_delay)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self.embedding_model is None:
            raise NotImplementedError(
                "No embedding_model configured for this backend. "
                "Pass embedding_model=<model-name> to enable embeddings."
            )
        response = await self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]


# Backwards-compatible alias
OpenAIBackend = OpenAICompatibleBackend


# ---------------------------------------------------------------------------
# Ollama backend (local LLM)
# ---------------------------------------------------------------------------

class OllamaBackend(LLMBackend):
    """
    Uses Ollama's REST API (http://localhost:11434 by default).
    Embedding via nomic-embed-text model.

    Start Ollama: ollama serve
    Pull models:  ollama pull llama3.2  &&  ollama pull nomic-embed-text
    """

    def __init__(
        self,
        model: str = "llama3.2",
        embedding_model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.embedding_model = embedding_model
        self.base_url = base_url.rstrip("/")

    async def complete(self, prompt: str, temperature: float = 0.7) -> str:
        import aiohttp

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/generate", json=payload
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                if "error" in data:
                    raise RuntimeError(f"Ollama error: {data['error']}")
                response = data.get("response", "")
                if not response:
                    logger.warning("Ollama returned empty response. Full payload: %s", data)
                return response

    async def _complete_json_raw(self, prompt: str, temperature: float = 0.3) -> str:
        """Use Ollama's native JSON mode (format=json) for reliable structured output."""
        import aiohttp

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature},
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/generate", json=payload
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                if "error" in data:
                    raise RuntimeError(f"Ollama error: {data['error']}")
                response = data.get("response", "")
                if not response:
                    logger.warning("Ollama returned empty response. Full payload: %s", data)
                return response

    async def complete_json(self, prompt: str, schema: Type[T], temperature: float = 0.3) -> T:
        """Override to use Ollama native JSON mode instead of the base implementation."""
        example = _schema_to_example(schema)
        json_prompt = (
            prompt
            + "\n\nOutput a JSON object with exactly these fields:\n"
            + json.dumps(example, indent=2)
        )
        raw = await self._complete_json_raw(json_prompt, temperature=temperature)
        raw = _strip_fences(raw)
        return schema.model_validate_json(raw)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import aiohttp

        embeddings = []
        async with aiohttp.ClientSession() as session:
            for text in texts:
                payload = {"model": self.embedding_model, "prompt": text}
                async with session.post(
                    f"{self.base_url}/api/embeddings", json=payload
                ) as resp:
                    data = await resp.json()
                    # Ollama >=0.5 uses "embeddings" (list of lists); older uses "embedding"
                    if "embeddings" in data:
                        embeddings.append(data["embeddings"][0])
                    else:
                        embeddings.append(data["embedding"])
        return embeddings


# ---------------------------------------------------------------------------
# Sentence-transformers backend (local embeddings, no server needed)
# ---------------------------------------------------------------------------

class SentenceTransformersBackend(LLMBackend):
    """
    Local embedding-only backend using sentence-transformers.
    Does NOT support complete() — pair with HybridBackend for that.

    Install: pip install sentence-transformers
    Good default model: "all-MiniLM-L6-v2" (fast, 384-dim)
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers") from e
        self._model = SentenceTransformer(model)

    async def complete(self, prompt: str, temperature: float = 0.7) -> str:
        raise NotImplementedError("SentenceTransformersBackend only supports embed(), not complete().")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import asyncio
        loop = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(None, lambda: self._model.encode(texts, convert_to_numpy=True))
        return [v.tolist() for v in vectors]


# ---------------------------------------------------------------------------
# Hybrid backend (split completions and embeddings across two backends)
# ---------------------------------------------------------------------------

class HybridBackend(LLMBackend):
    """
    Routes complete() to one backend and embed() to another.

    Useful when the completion API has no embedding endpoint:
        HybridBackend(
            completion=OpenAICompatibleBackend(model="...", base_url="https://llm.ai.e-infra.cz/v1"),
            embedding=SentenceTransformersBackend("all-MiniLM-L6-v2"),
        )
    """

    def __init__(self, completion: LLMBackend, embedding: LLMBackend) -> None:
        self._completion = completion
        self._embedding = embedding

    async def complete(self, prompt: str, temperature: float = 0.7) -> str:
        return await self._completion.complete(prompt, temperature)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._embedding.embed(texts)


# ---------------------------------------------------------------------------
# Mock backend (deterministic, no external calls)
# ---------------------------------------------------------------------------

class MockBackend(LLMBackend):
    """
    Returns deterministic placeholder responses.
    Useful for testing the simulation logic without LLM costs.
    """

    def __init__(self, always_forward: bool = True) -> None:
        self.always_forward = always_forward

    async def complete(self, prompt: str, temperature: float = 0.7) -> str:
        # Return minimal valid JSON if prompt asks for it
        if "forward" in prompt.lower() and "json" in prompt.lower():
            return json.dumps(
                {
                    "forward": self.always_forward,
                    "reasoning": "Mock agent always forwards." if self.always_forward else "Mock agent never forwards.",
                    "rewritten_content": None,
                }
            )
        return "[mock response]"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # Return zero vectors of dim 16
        return [[0.0] * 16 for _ in texts]


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def create_backend(
    backend: str = "openai",
    **kwargs: Any,
) -> LLMBackend:
    """
    backend: "openai" | "metacentrum" | "ollama" | "mock"
    kwargs: passed to the backend constructor

    MetaCentrum shortcut — set METACENTRUM_TOKEN env var, then:
        create_backend("metacentrum", model="meta-llama/Llama-3.3-70B-Instruct")
    """
    match backend:
        case "openai":
            kwargs.setdefault("model", "gpt-4o-mini")
            using_custom_url = "base_url" in kwargs or os.environ.get("OPENAI_BASE_URL")
            if using_custom_url:
                kwargs.setdefault("embedding_model", "nomic-embed-text-v1.5")
            else:
                kwargs.setdefault("embedding_model", "text-embedding-3-small")
            return OpenAICompatibleBackend(**kwargs)
        case "metacentrum":
            kwargs.setdefault("base_url", os.environ.get("METACENTRUM_BASE_URL", "https://llm.metacentrum.cz/v1"))
            kwargs.setdefault("api_key_env", "METACENTRUM_TOKEN")
            kwargs.setdefault("max_retries", 6)
            kwargs.setdefault("request_delay", 0.5)  # MetaCentrum throttles without 429
            return OpenAICompatibleBackend(**kwargs)
        case "ollama":
            return OllamaBackend(**kwargs)
        case "mock":
            return MockBackend(**kwargs)
        case _:
            raise ValueError(f"Unknown backend: {backend!r}")
