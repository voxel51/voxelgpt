"""
Link utils.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os
import re
import threading
import queue

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import (
    OpenAIModerationChain,
    RetrievalQA,
    RetrievalQAWithSourcesChain,
)
from langchain.chat_models import ChatOpenAI
from openai import Embedding


def get_cache():
    g = globals()
    if "_voxelgpt" not in g:
        g["_voxelgpt"] = {}

    return g["_voxelgpt"]


def get_openai_key():
    api_key = os.environ.get("OPENAI_API_KEY", None)
    if api_key is None:
        raise ValueError(
            "You must provide an OpenAI key by setting the OPENAI_API_KEY "
            "environment variable"
        )

    return api_key


def get_llm():
    return ChatOpenAI(
        openai_api_key=get_openai_key(),
        temperature=0,
        model_name="gpt-3.5-turbo",
    )


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
    except Exception as e:
        g.send(e)
    finally:
        g.close()


def query_retriever(retriever, prompt, sources=True):
    if not sources:
        qa = RetrievalQA.from_chain_type(
            llm=get_llm(),
            chain_type="stuff",
            retriever=retriever,
        )
        return qa.run([prompt]).strip()

    qa = RetrievalQAWithSourcesChain.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=retriever,
    )
    return qa({"question": prompt})


def stream_retriever(retriever, prompt, sources=True):
    g = ThreadedGenerator()
    threading.Thread(
        target=_retriever_thread, args=(g, retriever, sources, prompt)
    ).start()
    return g


def _retriever_thread(g, retriever, sources, prompt):
    try:
        llm = ChatOpenAI(
            openai_api_key=get_openai_key(),
            temperature=0,
            model_name="gpt-3.5-turbo",
            streaming=True,
            callbacks=[StreamingHandler(g)],
        )

        if sources:
            qa = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm, chain_type="stuff", retriever=retriever
            )
            response = qa({"question": prompt})
            g.send(response)
        else:
            qa = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=retriever
            )
            qa.run([prompt])
    except Exception as e:
        g.send(e)
    finally:
        g.close()


def embedding_function(queries):
    resp = Embedding.create(model="text-embedding-ada-002", input=queries)
    return [r["embedding"] for r in resp["data"]]


def get_embedding_function():
    return embedding_function


class FiftyOneModeration(OpenAIModerationChain):
    def _moderate(self, text: str, results: dict) -> str:
        return not results["flagged"]


def get_moderator():
    return FiftyOneModeration(openai_api_key=get_openai_key())


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
    """Splits raw tokens into whitespace-delimited words."""

    def __init__(self, gen, words=True):
        super().__init__()
        self.gen = gen
        self.words = words
        self._curr_word = ""

    def on_llm_new_token(self, token, **kwargs):
        if not self.words:
            self.gen.send(token)
            return

        # (chars, whitespace, chars, whitespace, ...)
        chunks = re.split("(\\s+)", token)

        self._curr_word += "".join(chunks[:2])
        if len(chunks) > 1:
            self.gen.send(self._curr_word)
            self._curr_word = "".join(chunks[2:])

    def on_llm_end(self, *args, **kwargs):
        if self._curr_word:
            self.gen.send(self._curr_word)
