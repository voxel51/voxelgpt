"""
Computer vision query dispatcher.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

from langchain.prompts import PromptTemplate

# pylint: disable=relative-beyond-top-level
from .utils import get_llm, stream_llm, get_cache


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = os.path.join(ROOT_DIR, "prompts")

CV_QUERY_TASK_RULES_PATH = os.path.join(
    PROMPTS_DIR, "computer_vision_query_task_rules.txt"
)


def _load_query_prefix():
    cache = get_cache()
    key = "computer_vision_query_prefix"
    if key not in cache:
        with open(CV_QUERY_TASK_RULES_PATH, "r") as f:
            cache[key] = f.read()
    return cache[key]


def _get_prompt(query):
    prefix = _load_query_prefix()
    return prefix + PromptTemplate(
        input_variables=["query"],
        template="Question: {query}\nAnswer:",
    ).format(query=query)


def run_computer_vision_query(query):
    prompt = _get_prompt(query)
    return get_llm().call_as_llm(prompt).strip()


def stream_computer_vision_query(query):
    prompt = _get_prompt(query)
    for content in stream_llm(prompt):
        yield content
