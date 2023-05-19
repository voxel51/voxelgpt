"""
Computer vision query dispatcher.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

from langchain.prompts import PromptTemplate

# pylint: disable=relative-beyond-top-level
from .utils import get_llm


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = os.path.join(ROOT_DIR, "prompts")

CV_QUERY_TASK_RULES_PATH = os.path.join(
    PROMPTS_DIR, "computer_vision_query_task_rules.txt"
)


def load_query_prefix():
    with open(CV_QUERY_TASK_RULES_PATH, "r") as f:
        return f.read()


def run_computer_vision_query(query):
    prefix = load_query_prefix()
    prompt = prefix + PromptTemplate(
        input_variables=["query"],
        template="Question: {query}\nAnswer:",
    ).format(query=query)

    response = get_llm().call_as_llm(prompt)
    return response.strip()
