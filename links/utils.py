"""
Link utils.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

import chromadb
from chromadb.utils import embedding_functions
from langchain.chat_models import ChatOpenAI


client = None
llm = None
embedding_function = None


def get_openai_key():
    api_key = os.environ.get("OPENAI_API_KEY", None)
    if api_key is None:
        raise ValueError(
            "You must provide an OpenAI key by setting the OPENAI_API_KEY "
            "environment variable"
        )

    return api_key


def get_chromadb_client():
    global client

    if client is None:
        client = chromadb.Client()

    return client


def get_llm():
    global llm

    if llm is None:
        llm = ChatOpenAI(
            openai_api_key=get_openai_key(),
            temperature=0,
            model_name="gpt-3.5-turbo",
        )

    return llm


def get_embedding_function():
    global embedding_function

    if embedding_function is None:
        embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=get_openai_key(),
            model_name="text-embedding-ada-002",
        )

    return embedding_function