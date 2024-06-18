"""
Chain for introspection on VoxelGPT's capabilities.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

from langchain_core.runnables import RunnableLambda

# pylint: disable=relative-beyond-top-level
from .utils import (
    PROMPTS_DIR,
    _build_chat_chain,
    gpt_4o,
    stream_runnable,
    get_prompt_from,
)

VOXELGPT_INFO_PATH = os.path.join(PROMPTS_DIR, "help_dynamic.txt")


def stream_introspection_query(query):
    prompt = get_prompt_from(VOXELGPT_INFO_PATH).format(question=query)
    chain = _build_chat_chain(gpt_4o, prompt=prompt)

    def func_streaming(info):
        query = info["query"]
        for chunk in chain.stream({"messages": [("user", query)]}):
            yield chunk

    runnable_streaming = RunnableLambda(func_streaming)

    for content in stream_runnable(runnable_streaming, {"query": query}):
        if isinstance(content, Exception):
            raise content
        yield content.content


def run_introspection_query(query):
    prompt = get_prompt_from(VOXELGPT_INFO_PATH).format(question=query)
    chain = _build_chat_chain(gpt_4o, prompt=prompt)

    def func(info):
        query = info["query"]
        response = chain.invoke({"messages": [("user", query)]}).content
        return {"input": query, "output": response}

    runnable = RunnableLambda(func)
    return runnable.invoke({"query": query})["output"]
