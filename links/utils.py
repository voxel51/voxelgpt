"""
Link utils.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import OpenAIModerationChain
from langchain.chat_models import ChatOpenAI
from openai import Embedding


def get_openai_key():
    api_key = os.environ.get("OPENAI_API_KEY", None)
    if api_key is None:
        raise ValueError(
            "You must provide an OpenAI key by setting the OPENAI_API_KEY "
            "environment variable"
        )

    return api_key


def get_llm(streaming=False):
    key = "llm_streaming" if streaming else "llm"
    cache = get_cache()
    if key not in cache:
        if streaming:
            cache[key] = ChatOpenAI(
                openai_api_key=get_openai_key(),
                temperature=0,
                model_name="gpt-3.5-turbo",
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()],
            )
        else:
            cache[key] = ChatOpenAI(
                openai_api_key=get_openai_key(),
                temperature=0,
                model_name="gpt-3.5-turbo",
            )

    return cache[key]


def embedding_function(queries):
    resp = Embedding.create(model="text-embedding-ada-002", input=queries)
    return [r["embedding"] for r in resp["data"]]


def get_embedding_function():
    cache = get_cache()
    if "embedding_function" not in cache:
        cache["embedding_function"] = embedding_function
    return cache["embedding_function"]


class FiftyOneModeration(OpenAIModerationChain):
    def _moderate(self, text: str, results: dict) -> str:
        return not results["flagged"]


def get_moderator():
    cache = get_cache()
    if "moderator" not in cache:
        cache["moderator"] = FiftyOneModeration(
            openai_api_key=get_openai_key()
        )

    return cache["moderator"]


def get_cache():
    g = globals()
    if "_voxelgpt" not in g:
        g["_voxelgpt"] = {}

    return g["_voxelgpt"]
