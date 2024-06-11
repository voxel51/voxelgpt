"""
FiftyOne docs query dispatcher.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os
import requests

import numpy as np

from langchain_core.prompts import (
    FewShotPromptTemplate,
    PromptTemplate,
)
from langchain_core.runnables import RunnableLambda

# pylint: disable=relative-beyond-top-level
from .utils import (
    get_prompt_from,
    PROMPTS_DIR,
    stream_runnable,
    gpt_4o,
    embedding_model,
)

PROTECT_MAPS = [
    ("{source}", "<SOURCE>"),
    ("{page_content}", "<CONTENT>"),
    ("{", "LEFT_BRACE"),
    ("}", "RIGHT_BRACE"),
]

DOCS_QA_PATH = os.path.join(PROMPTS_DIR, "docs_qa_retrieval.txt")
DOCS_QA_PROMPT_TEMPLATE = get_prompt_from(DOCS_QA_PATH)


def unprotect_text(text):
    for k, v in PROTECT_MAPS:
        text = text.replace(v, k)
    return text


def protect_text(text):
    for k, v in PROTECT_MAPS:
        text = text.replace(k, v)
    return text


def _build_docs_qa_prompt(query, docs):
    example_template = PromptTemplate(
        template="Content: {page_content}\nSource: {source}",
        input_variables=["page_content", "source"],
    )

    examples = [
        {
            "page_content": protect_text(doc[0]),
            "source": doc[1],
        }
        for doc in docs
    ]

    summaries = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_template,
        prefix="",
        suffix="",
        input_variables=[],
    ).format()

    prompt = DOCS_QA_PROMPT_TEMPLATE.format(
        question=query,
        summaries=summaries,
    )
    return unprotect_text(prompt)


def _get_documents(query):
    query_vector = embedding_model.embed_query(query)
    query_vector = [str(np.round(qv, 8)) for qv in query_vector]
    response = requests.get(
        "http://127.0.0.1:5000/retrieve", params={"query": query_vector}
    )
    response = response.json()["results"]
    return response


def docs_func(info):
    query = info["query"]
    documents = _get_documents(query)
    prompt = _build_docs_qa_prompt(query, documents)
    response = gpt_4o.invoke(prompt)

    return {"input": query, "output": response.content}


def docs_func_streaming(info):
    query = info["query"]
    documents = _get_documents(query)
    prompt = _build_docs_qa_prompt(query, documents)
    for chunk in gpt_4o.stream(prompt):
        yield chunk


def run_docs_query(query):
    docs_runnable = RunnableLambda(docs_func)
    return docs_runnable.invoke({"query": query})["output"]


def stream_docs_query(query):
    docs_runnable_streaming = RunnableLambda(docs_func_streaming)
    for content in stream_runnable(docs_runnable_streaming, {"query": query}):
        if isinstance(content, Exception):
            raise content
        yield content.content
