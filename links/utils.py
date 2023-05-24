"""
Link utils.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

import chromadb
from chromadb.utils import embedding_functions
from langchain.chains import OpenAIModerationChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings


def get_openai_key():
    api_key = os.environ.get("OPENAI_API_KEY", None)
    if api_key is None:
        raise ValueError(
            "You must provide an OpenAI key by setting the OPENAI_API_KEY "
            "environment variable"
        )

    return api_key


def get_chromadb_client():
    cache = get_cache()
    if "chromadb_client" not in cache:
        cache["chromadb_client"] = chromadb.Client()

    return cache["chromadb_client"]


def get_llm():
    cache = get_cache()
    if "llm" not in cache:
        cache["llm"] = ChatOpenAI(
            openai_api_key=get_openai_key(),
            temperature=0,
            model_name="gpt-3.5-turbo",
        )

    return cache["llm"]


def get_embedding_function():
    cache = get_cache()
    if "embedding_function" not in cache:
        cache[
            "embedding_function"
        ] = embedding_functions.OpenAIEmbeddingFunction(
            api_key=get_openai_key(),
            model_name="text-embedding-ada-002",
        )

    return cache["embedding_function"]


def get_embedding_model():
    cache = get_cache()
    if "embedding_model" not in cache:
        cache["embedding_model"] = OpenAIEmbeddings(
            openai_api_key=get_openai_key()
        )

    return cache["embedding_model"]


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
