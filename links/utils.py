"""
Link utils.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os
import threading
import queue

from langchain.callbacks.base import BaseCallbackHandler
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


def get_llm():
    cache = get_cache()
    if "llm" not in cache:
        cache["llm"] = ChatOpenAI(
            openai_api_key=get_openai_key(),
            temperature=0,
            model_name="gpt-3.5-turbo",
        )

    return cache["llm"]


def stream_llm(prompt):
    g = ThreadedGenerator()
    threading.Thread(target=_llm_thread, args=(g, prompt)).start()
    return g


def _llm_thread(g, prompt):
    try:
        llm = ChatOpenAI(
            openai_api_key=get_openai_key(),
            temperature=0,
            model_name="gpt-3.5-turbo",
            streaming=True,
            callbacks=[StreamingHandler(g)],
        )
        llm.call_as_llm(prompt)
    finally:
        g.close()


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


class ThreadedGenerator(object):
    def __init__(self):
        self.queue = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is StopIteration:
            raise item

        return item

    def send(self, data):
        self.queue.put(data)

    def close(self):
        self.queue.put(StopIteration)


class StreamingHandler(BaseCallbackHandler):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def on_llm_new_token(self, token, **kwargs):
        self.gen.send(token)
