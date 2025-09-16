import asyncio
import base64
import copy
import json
import os
import re
import struct
from functools import lru_cache
from typing import List, Dict, Callable, Any, Union, Optional
import aioboto3
import aiohttp
import numpy as np
import ollama
import torch
import time
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    Timeout,
    AsyncAzureOpenAI,
)
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from transformers import AutoTokenizer, AutoModelForCausalLM

from .utils import (
    wrap_embedding_func_with_attrs,
    locate_json_string_body_from_string,
    safe_unicode_decode,
    logger,
)

import sys

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def openai_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
    **kwargs,
) -> str:
    time.sleep(2)
    openai_async_client = AsyncOpenAI(base_url=base_url) if base_url else AsyncOpenAI()
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    logger.debug("===== Query Input to LLM =====")
    logger.debug(f"Query: {prompt}")
    logger.debug(f"System prompt: {system_prompt}")
    logger.debug("Full context:")
    SEM = asyncio.Semaphore(5)
    async with SEM:
        if "response_format" in kwargs:
            response = await openai_async_client.beta.chat.completions.parse(
                model=model, messages=messages, **kwargs
            )
        else:
            response = await openai_async_client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )

        if hasattr(response, "__aiter__"):

            async def inner():
                async for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content is None:
                        continue
                    if r"\u" in content:
                        content = safe_unicode_decode(content.encode("utf-8"))
                    yield content

            return inner()
        else:
            content = response.choices[0].message.content
            if r"\u" in content:
                content = safe_unicode_decode(content.encode("utf-8"))
            return content


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def azure_openai_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    base_url=None,
    api_key=None,
    api_version=None,
    **kwargs,
):
    if api_key:
        os.environ["AZURE_OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["AZURE_OPENAI_ENDPOINT"] = base_url
    if api_version:
        os.environ["AZURE_OPENAI_API_VERSION"] = api_version

    openai_async_client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    if prompt is not None:
        messages.append({"role": "user", "content": prompt})

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )
    content = response.choices[0].message.content

    return content


@lru_cache(maxsize=1)
def initialize_hf_model(model_name):
    hf_tokenizer = AutoTokenizer.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True
    )
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    return hf_model, hf_tokenizer


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def hf_model_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    **kwargs,
) -> str:
    model_name = model
    hf_model, hf_tokenizer = initialize_hf_model(model_name)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    kwargs.pop("hashing_kv", None)
    input_prompt = ""
    try:
        input_prompt = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        try:
            ori_message = copy.deepcopy(messages)
            if messages[0]["role"] == "system":
                messages[1]["content"] = (
                    "<system>"
                    + messages[0]["content"]
                    + "</system>\n"
                    + messages[1]["content"]
                )
                messages = messages[1:]
                input_prompt = hf_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        except Exception:
            len_message = len(ori_message)
            for msgid in range(len_message):
                input_prompt = (
                    input_prompt
                    + "<"
                    + ori_message[msgid]["role"]
                    + ">"
                    + ori_message[msgid]["content"]
                    + "</"
                    + ori_message[msgid]["role"]
                    + ">\n"
                )

    input_ids = hf_tokenizer(
        input_prompt, return_tensors="pt", padding=True, truncation=True
    ).to("cuda")
    inputs = {k: v.to(hf_model.device) for k, v in input_ids.items()}
    output = hf_model.generate(
        **input_ids, max_new_tokens=512, num_return_sequences=1, early_stopping=True
    )
    response_text = hf_tokenizer.decode(
        output[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
    )

    return response_text


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def ollama_model_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    stream = True if kwargs.get("stream") else False
    kwargs.pop("max_tokens", None)
    host = kwargs.pop("host", None)
    timeout = kwargs.pop("timeout", None)
    kwargs.pop("hashing_kv", None)
    ollama_client = ollama.AsyncClient(host=host, timeout=timeout)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = await ollama_client.chat(model=model, messages=messages, **kwargs)
    if stream:
        """cannot cache stream response"""

        async def inner():
            async for chunk in response:
                yield chunk["message"]["content"]

        return inner()
    else:
        return response["message"]["content"]


class GPTKeywordExtractionFormat(BaseModel):
    high_level_keywords: List[str]
    low_level_keywords: List[str]


async def openai_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> Union[str, AsyncIterator[str]]:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = "json"
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def siliconflow_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat
    return await openai_complete_if_cache(
        "deepseek-ai/DeepSeek-V2.5",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def azure_openai_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    result = await azure_openai_complete_if_cache(
        "conversation-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )
    if keyword_extraction:  # TODO: use JSON API
        return locate_json_string_body_from_string(result)
    return result


async def hf_model_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    result = await hf_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )
    if keyword_extraction:  # TODO: use JSON API
        return locate_json_string_body_from_string(result)
    return result


async def ollama_model_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> Union[str, AsyncIterator[str]]:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["format"] = "json"
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    model_name = "qwen3:32b"
    return await ollama_model_if_cache(
        model_name,
        prompt + " /no_think",
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=2560, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def openai_embedding(
    texts: list[str],
    model: str = "Qwen/Qwen3-Embedding-4B",
    base_url="",
    api_key="",
) -> np.ndarray:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    # Allow api_key parameter to override environment variable
    token = api_key or os.getenv("OPENAI_API_KEY", "")

    url = f'{os.environ["OPENAI_BASE_URL"]}/embeddings'
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "input": texts,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data, headers=headers) as response:
            response_json = await response.json()

    data_list = response_json.get("data", [])
    return np.array([dp["embedding"] for dp in data_list])


async def fetch_data(url, headers, data):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            response_json = await response.json()
            data_list = response_json.get("data", [])
            return data_list


async def jina_embedding(
    texts: list[str],
    dimensions: int = 1024,
    late_chunking: bool = False,
    base_url: str = None,
    api_key: str = None,
) -> np.ndarray:
    if api_key:
        os.environ["JINA_API_KEY"] = api_key
    url = "https://api.jina.ai/v1/embeddings" if not base_url else base_url
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['JINA_API_KEY']}",
    }
    data = {
        "model": "jina-embeddings-v3",
        "normalized": True,
        "embedding_type": "float",
        "dimensions": f"{dimensions}",
        "late_chunking": late_chunking,
        "input": texts,
    }
    data_list = await fetch_data(url, headers, data)
    return np.array([dp["embedding"] for dp in data_list])


@wrap_embedding_func_with_attrs(embedding_dim=2560, max_token_size=512)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def nvidia_openai_embedding(
    texts: list[str],
    model: str = "nvidia/llama-3.2-nv-embedqa-1b-v1",
    base_url: str = "https://integrate.api.nvidia.com/v1",
    api_key: str = None,
    input_type: str = "passage",
    trunc: str = "NONE",
    encode: str = "float",
) -> np.ndarray:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    openai_async_client = AsyncOpenAI(base_url=base_url) if base_url else AsyncOpenAI()
    response = await openai_async_client.embeddings.create(
        model=model,
        input=texts,
        encoding_format=encode,
        extra_body={"input_type": input_type, "truncate": trunc},
    )
    return np.array([dp.embedding for dp in response.data])


@wrap_embedding_func_with_attrs(embedding_dim=2560, max_token_size=8191)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def azure_openai_embedding(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str = None,
    api_key: str = None,
    api_version: str = None,
) -> np.ndarray:
    if api_key:
        os.environ["AZURE_OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["AZURE_OPENAI_ENDPOINT"] = base_url
    if api_version:
        os.environ["AZURE_OPENAI_API_VERSION"] = api_version

    openai_async_client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )

    response = await openai_async_client.embeddings.create(
        model=model, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def siliconcloud_embedding(
    texts: list[str],
    model: str = "Qwen/Qwen3-Embedding-4B",
    base_url: str = "https://api.siliconflow.cn/v1/embeddings",
    max_token_size: int = 8192,
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


async def hf_embedding(texts: list[str], tokenizer, embed_model) -> np.ndarray:
    device = next(embed_model.parameters()).device
    input_ids = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).input_ids.to(device)
    with torch.no_grad():
        outputs = embed_model(input_ids)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    if embeddings.dtype == torch.bfloat16:
        return embeddings.detach().to(torch.float32).cpu().numpy()
    else:
        return embeddings.detach().cpu().numpy()


async def ollama_embedding(texts: list[str], embed_model, **kwargs) -> np.ndarray:
    """
    Deprecated in favor of `embed`.
    """
    embed_text = []
    ollama_client = ollama.Client(**kwargs)
    for text in texts:
        data = ollama_client.embeddings(model=embed_model, prompt=text)
        embed_text.append(data["embedding"])

    return embed_text


async def ollama_embed(texts: list[str], embed_model, **kwargs) -> np.ndarray:
    ollama_client = ollama.Client(**kwargs)
    data = ollama_client.embed(model=embed_model, input=texts)
    return data["embeddings"]


class Model(BaseModel):
    """
    This is a Pydantic model class named 'Model' that is used to define a custom language model.

    Attributes:
        gen_func (Callable[[Any], str]): A callable function that generates the response from the language model.
            The function should take any argument and return a string.
        kwargs (Dict[str, Any]): A dictionary that contains the arguments to pass to the callable function.
            This could include parameters such as the model name, API key, etc.

    Example usage:
        Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_1"]})

    In this example, 'openai_complete_if_cache' is the callable function that generates the response from the OpenAI model.
    The 'kwargs' dictionary contains the model name and API key to be passed to the function.
    """

    gen_func: Callable[[Any], str] = Field(
        ...,
        description="A function that generates the response from the llm. The response must be a string",
    )
    kwargs: Dict[str, Any] = Field(
        ...,
        description="The arguments to pass to the callable function. Eg. the api key, model name, etc",
    )

    class Config:
        arbitrary_types_allowed = True


class MultiModel:
    """
    Distributes the load across multiple language models. Useful for circumventing low rate limits with certain api providers especially if you are on the free tier.
    Could also be used for spliting across diffrent models or providers.

    Attributes:
        models (List[Model]): A list of language models to be used.

    Usage example:
        ```python
        models = [
            Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_1"]}),
            Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_2"]}),
            Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_3"]}),
            Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_4"]}),
            Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_5"]}),
        ]
        multi_model = MultiModel(models)
        rag = LightRAG(
            llm_model_func=multi_model.llm_model_func
            / ..other args
            )
        ```
    """

    def __init__(self, models: List[Model]):
        self._models = models
        self._current_model = 0

    def _next_model(self):
        self._current_model = (self._current_model + 1) % len(self._models)
        return self._models[self._current_model]

    async def llm_model_func(
        self, prompt, system_prompt=None, history_messages=[], **kwargs
    ) -> str:
        kwargs.pop("model", None)
        kwargs.pop("keyword_extraction", None)
        kwargs.pop("mode", None)
        next_model = self._next_model()
        args = dict(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
            **next_model.kwargs,
        )

        return await next_model.gen_func(**args)


if __name__ == "__main__":
    import asyncio

    async def main():
        result = await gpt_4o_mini_complete("How are you?")
        print(result)

    asyncio.run(main())
