"""
Effective query generator.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

# pylint: disable=relative-beyond-top-level
from .utils import get_llm


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = os.path.join(ROOT_DIR, "prompts")

EFFECTIVE_PROMPT_GENERATOR_PREFIX_PATH = os.path.join(
    PROMPTS_DIR, "effective_prompt_generator_prefix.txt"
)


def load_effective_prompt_prefix_template():
    with open(EFFECTIVE_PROMPT_GENERATOR_PREFIX_PATH, "r") as f:
        return f.read()


def format_chat_history(chat_history):
    return "\n".join(chat_history) + "\n"


def generate_dataset_view_prompt(chat_history):
    prompt = load_effective_prompt_prefix_template()
    prompt += format_chat_history(chat_history)
    prompt += "Effective prompt: "
    return prompt


def generate_effective_query(chat_history):
    prompt = generate_dataset_view_prompt(chat_history)
    response = get_llm().call_as_llm(prompt)
    return response.strip()
