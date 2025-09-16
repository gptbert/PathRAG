import asyncio
import base64
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

from .utils import safe_unicode_decode, logger


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


async def siliconcloud_embedding(
    texts: list[str],
    model: str = "Qwen/Qwen3-Embedding-4B",
    base_url: str = "https://api.siliconflow.cn/v1/embeddings",
    max_token_size: int = 512,
    api_key: str = None,
) -> np.ndarray:
    if api_key and not api_key.startswith("Bearer "):
        api_key = "Bearer " + api_key

    headers = {"Authorization": api_key, "Content-Type": "application/json"}

    truncate_texts = [text[0:max_token_size] for text in texts]

    payload = {"model": model, "input": truncate_texts, "encoding_format": "base64"}

    base64_strings = []
    async with aiohttp.ClientSession() as session:
        async with session.post(base_url, headers=headers, json=payload) as response:
            content = await response.json()
            if "code" in content:
                raise ValueError(content)
            base64_strings = [item["embedding"] for item in content["data"]]

    embeddings = []
    for string in base64_strings:
        decode_bytes = base64.b64decode(string)
        n = len(decode_bytes) // 4
        float_array = struct.unpack("<" + "f" * n, decode_bytes)
        embeddings.append(float_array)
    return np.array(embeddings)


if __name__ == "__main__":

    async def main():
        result = await siliconflow_complete("How are you?")
        print(result)

    asyncio.run(main())
