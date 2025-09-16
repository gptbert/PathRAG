import asyncio
import base64
import hashlib
import os
import struct
from collections.abc import AsyncIterator
from typing import Any, Dict, List, Optional, Union

import aiohttp
import numpy as np
from openai import AsyncOpenAI, APIConnectionError, RateLimitError, Timeout
from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .utils import wrap_embedding_func_with_attrs, safe_unicode_decode, logger


class GPTKeywordExtractionFormat(BaseModel):
    high_level_keywords: List[str]
    low_level_keywords: List[str]


def _build_messages(
    prompt: str,
    system_prompt: Optional[str],
    history_messages: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    return messages


def _get_async_client(base_url: Optional[str], api_key: Optional[str]) -> AsyncOpenAI:
    client_kwargs: Dict[str, Any] = {}
    env_base_url = (
        base_url or os.getenv("SILICONFLOW_BASE_URL")
    )
    if env_base_url:
        client_kwargs["base_url"] = env_base_url
    if api_key or os.getenv("SILICONFLOW_API_KEY"):
        client_kwargs["api_key"] = api_key or os.getenv("SILICONFLOW_API_KEY")
    return AsyncOpenAI(**client_kwargs)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def siliconflow_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict[str, str]]] = None,
    keyword_extraction: bool = False,
    *,
    model: str = "deepseek-ai/DeepSeek-V2.5",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    """Call the SiliconFlow chat completion endpoint using its OpenAI-compatible API."""

    history_messages = history_messages or []
    kwargs.pop("hashing_kv", None)
    kwargs.pop("mode", None)
    kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat

    messages = _build_messages(prompt, system_prompt, history_messages)
    client = _get_async_client(base_url, api_key)

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )

    if hasattr(response, "__aiter__"):

        async def inner():
            async for chunk in response:
                content = chunk.choices[0].delta.content
                if not content:
                    continue
                if r"\u" in content:
                    content = safe_unicode_decode(content.encode("utf-8"))
                yield content

        return inner()

    content = response.choices[0].message.content
    if content and r"\u" in content:
        content = safe_unicode_decode(content.encode("utf-8"))

    logger.debug("===== SiliconFlow LLM Response =====")
    logger.debug(content)
    return content or ""


def _deterministic_embedding(text: str, dim: int) -> np.ndarray:
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


@wrap_embedding_func_with_attrs(embedding_dim=2560, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def siliconcloud_embedding(
    texts: List[str],
    model: str = "Qwen/Qwen3-Embedding-4B",
    base_url: str = "https://api.siliconflow.cn/v1/embeddings",
    max_token_size: int = 8192,
    api_key: Optional[str] = None,
) -> np.ndarray:
    dim = siliconcloud_embedding.embedding_dim  # type: ignore[attr-defined]
    if not texts:
        return np.empty((0, dim))

    truncate_texts = [text[:max_token_size] for text in texts]

    def local_embeddings() -> np.ndarray:
        logger.warning(
            "SiliconFlow embedding API unavailable; falling back to deterministic embeddings."
        )
        return np.stack([_deterministic_embedding(text, dim) for text in truncate_texts])

    token = (
        api_key
        or os.getenv("SILICONFLOW_API_KEY")
    )

    if not token:
        return local_embeddings()

    if not token.startswith("Bearer "):
        token = "Bearer " + token

    headers = {"Content-Type": "application/json", "Authorization": token}
    payload = {"model": model, "input": truncate_texts, "encoding_format": "base64"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(base_url, headers=headers, json=payload) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: {await response.text()}")
                content = await response.json()
    except Exception as exc:
        logger.warning(f"SiliconFlow embedding request failed: {exc}")
        return local_embeddings()

    data = content.get("data") if isinstance(content, dict) else None
    if not data:
        logger.warning(f"Unexpected embedding response format: {content}")
        return local_embeddings()

    base64_strings = [item.get("embedding") for item in data if isinstance(item, dict)]
    if not base64_strings or any(not isinstance(b, str) for b in base64_strings):
        logger.warning(f"Missing embedding vectors in response: {content}")
        return local_embeddings()

    embeddings = []
    for string in base64_strings:
        decode_bytes = base64.b64decode(string)
        n = len(decode_bytes) // 4
        float_array = struct.unpack("<" + "f" * n, decode_bytes)
        embeddings.append(float_array)

    if len(embeddings) != len(truncate_texts):
        logger.warning(
            f"Embedding count mismatch ({len(embeddings)} vs {len(truncate_texts)}); using local fallback."
        )
        return local_embeddings()

    return np.array(embeddings)


if __name__ == "__main__":

    async def main():
        result = await siliconflow_complete("How are you?")
        print(result)

    asyncio.run(main())
