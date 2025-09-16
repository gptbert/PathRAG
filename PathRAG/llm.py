import asyncio
import base64
import os
import struct
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np
from openai import AsyncOpenAI, APIConnectionError, RateLimitError, Timeout
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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
    if base_url or os.getenv("OPENAI_BASE_URL"):
        client_kwargs["base_url"] = base_url or os.getenv("OPENAI_BASE_URL")
    if api_key or os.getenv("OPENAI_API_KEY"):
        client_kwargs["api_key"] = api_key or os.getenv("OPENAI_API_KEY")
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
) -> str:
    """Call the SiliconFlow chat completion endpoint using its OpenAI-compatible API."""

    history_messages = history_messages or []
    kwargs.pop("hashing_kv", None)
    kwargs.pop("mode", None)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat

    messages = _build_messages(prompt, system_prompt, history_messages)
    client = _get_async_client(base_url, api_key)

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )

    content = response.choices[0].message.content
    if content and r"\u" in content:
        content = safe_unicode_decode(content.encode("utf-8"))

    logger.debug("===== SiliconFlow LLM Response =====")
    logger.debug(content)
    return content or ""


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
    """Create embeddings via SiliconFlow's embedding API."""

    if api_key and not api_key.startswith("Bearer "):
        api_key = "Bearer " + api_key

    headers = {"Authorization": api_key or os.getenv("SILICONFLOW_TOKEN", ""), "Content-Type": "application/json"}
    truncate_texts = [text[:max_token_size] for text in texts]
    payload = {"model": model, "input": truncate_texts, "encoding_format": "base64"}

    async with aiohttp.ClientSession() as session:
        async with session.post(base_url, headers=headers, json=payload) as response:
            content = await response.json()
            if "code" in content:
                raise ValueError(content)
            base64_strings = [item["embedding"] for item in content["data"]]

    embeddings = []
    for encoded in base64_strings:
        decode_bytes = base64.b64decode(encoded)
        n = len(decode_bytes) // 4
        float_array = struct.unpack("<" + "f" * n, decode_bytes)
        embeddings.append(float_array)
    return np.array(embeddings)


if __name__ == "__main__":
    async def main():
        result = await siliconflow_complete("How are you?")
        print(result)

    asyncio.run(main())
