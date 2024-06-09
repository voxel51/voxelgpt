"""
Computer vision query dispatcher.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

from langchain_core.runnables import RunnableLambda

# pylint: disable=relative-beyond-top-level
from .utils import PROMPTS_DIR, _build_chat_chain, gpt_4o, stream_runnable

CV_QA_PATH = os.path.join(PROMPTS_DIR, "computer_vision_response.txt")
cv_chain = _build_chat_chain(gpt_4o, template_path=CV_QA_PATH)


def cv_func(info):
    query = info["query"]
    response = cv_chain.invoke({"messages": [("user", query)]}).content
    return {"input": query, "output": response}


def cv_func_streaming(info):
    query = info["query"]
    for chunk in cv_chain.stream({"messages": [("user", query)]}):
        yield chunk


def stream_computer_vision_query(query):
    cv_runnable_streaming = RunnableLambda(cv_func_streaming)
    for content in stream_runnable(cv_runnable_streaming, {"query": query}):
        if isinstance(content, Exception):
            raise content
        yield content.content


def run_computer_vision_query(query):
    cv_runnable = RunnableLambda(cv_func)
    return cv_runnable.invoke({"query": query})["output"]
