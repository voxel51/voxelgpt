"""
Link utils.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os
import re
import threading
import queue

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
)
from langchain_openai import ChatOpenAI

gpt_3_5 = ChatOpenAI(model="gpt-3.5-turbo")
gpt_4o = ChatOpenAI(model="gpt-4o")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = os.path.join(ROOT_DIR, "prompts")

REPLACEMENT_MAPPING = {
    'F("$IMAGE_AREA")': 'F("$IMAGE_WIDTH") * F("$IMAGE_HEIGHT")',
    'F("ABS_BBOX_WIDTH")': 'F("REL_BBOX_WIIDTH") * F("$IMAGE_WIDTH")',
    'F("ABS_BBOX_HEIGHT")': 'F("REL_BBOX_HEIGHT") * F("$IMAGE_HEIGHT")',
    'F("ABS_BBOX_AREA")': 'F("REL_BBOX_AREA") * F("$IMAGE_AREA")',
    "F('ABS_BBOX_AREA')": 'F("REL_BBOX_AREA") * F("$IMAGE_AREA")',
    "F('BBOX_VOLUME')": 'F("BBOX_VOLUME")',
    'F("BBOX_VOLUME")': 'F("BBOX_X") * F("BBOX_Y") * F("BBOX_Z")',
    'F("BBOX_X")': 'abs(F("dimensions")[0])',
    "F('BBOX_Y')": 'abs(F("dimensions")[1])',
    'F("BBOX_Z")': 'abs(F("dimensions")[2])',
    'F("$IMAGE_WIDTH")': 'F("$metadata.width")',
    "F('IMAGE_WIDTH')": 'F("$metadata.width")',
    'F("REL_BBOX_AREA")': 'F("bounding_box")[2] * F("bounding_box")[3]',
    "F('REL_BBOX_AREA')": 'F("bounding_box")[2] * F("bounding_box")[3]',
    'F("REL_BBOX_WIIDTH")': 'F("bounding_box")[2]',
    "F('REL_BBOX_WIIDTH')": 'F("bounding_box")[2]',
    'F("REL_BBOX_HEIGHT")': 'F("bounding_box")[3]',
    "F('REL_BBOX_HEIGHT')": 'F("bounding_box")[3]',
    'F("IMAGE_AREA")': 'F("metadata.width") * F("metadata.height")',
    "F('IMAGE_AREA')": 'F("metadata.width") * F("metadata.height")',
    'F("IMAGE_WIDTH")': 'F("metadata.width")',
    "F('IMAGE_WIDTH')": 'F("metadata.width")',
    'F("IMAGE_HEIGHT")': 'F("metadata.height")',
    "F('IMAGE_HEIGHT')": 'F("metadata.height")',
}


def get_prompt_from(path):
    with open(path, "r") as f:
        return f.read()


def _make_replacements(expr):
    for key, value in REPLACEMENT_MAPPING.items():
        expr = expr.replace(key, value)
    return expr


def _build_custom_chain(model, template_path):
    prompt = get_prompt_from(template_path)
    chain = PromptTemplate.from_template(prompt) | model | StrOutputParser()
    return chain


def _build_chat_chain(
    model, output_type=None, template_path=None, prompt=None
):
    if template_path:
        prompt = get_prompt_from(template_path)
    curr_model = model
    if output_type:
        curr_model = curr_model.with_structured_output(output_type)
    chain = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    prompt,
                ),
                ("placeholder", "{messages}"),
            ]
        )
        | curr_model
    )
    return chain


def _build_agent_executor_chain(model, tools, template_path):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", get_prompt_from(template_path)),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(model, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
    )
    return agent_executor


def _build_runnable_thread(runnable, info):
    def _runnable_thread(runnable, info, q):
        for chunk in runnable.stream(info):
            q.put(chunk)
        q.put(None)

    q = queue.Queue()
    thread = threading.Thread(
        target=_runnable_thread, args=(runnable, info, q)
    )
    thread.start()
    return q


def _get_runnable_thread_output(q):
    while True:
        chunk = q.get()
        if chunk is None:
            break
        yield chunk


def stream_runnable(runnable, info):
    q = _build_runnable_thread(runnable, info)
    return _get_runnable_thread_output(q)


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
    """Pass `words=True` to split tokens into whitespace-delimited words."""

    def __init__(self, gen, words=False):
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


def _format_filter_expression(filter_expr):
    if filter_expr[0] == '"' and filter_expr[-1] == '"':
        filter_expr = filter_expr[1:-1]
    elif filter_expr[0] == "'" and filter_expr[-1] == "'":
        filter_expr = filter_expr[1:-1]

    filter_expr = filter_expr.replace("`", "")
    filter_expr = _make_replacements(filter_expr)
    filter_expr = _replace_threshold_if_necessary(filter_expr)
    return filter_expr


def _replace_threshold_if_necessary(filter_expr):
    if "threshold" in filter_expr:
        if ">" in filter_expr:
            filter_expr = filter_expr.replace("threshold", "0.9")
        else:
            filter_expr = filter_expr.replace("threshold", "0.1")
    return filter_expr
