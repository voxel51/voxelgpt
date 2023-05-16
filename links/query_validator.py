"""
Query validator.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

from langchain.prompts import PromptTemplate, FewShotPromptTemplate
import pandas as pd

# pylint: disable=relative-beyond-top-level
from .utils import get_llm, get_moderator


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")
PROMPTS_DIR = os.path.join(ROOT_DIR, "prompts")

QUERY_VALIDATION_EXAMPLES_PATH = os.path.join(
    EXAMPLES_DIR, "fiftyone_query_validation_examples.csv"
)
CONFUSED_TASK_RULES_PATH = os.path.join(PROMPTS_DIR, "confused_task_rules.txt")


def load_query_validator_prefix():
    with open(CONFUSED_TASK_RULES_PATH, "r") as f:
        return f.read()


def get_query_validation_examples():
    df = pd.read_csv(QUERY_VALIDATION_EXAMPLES_PATH)
    examples = []

    for _, row in df.iterrows():
        example = {"input": row.input, "is_valid": row.is_valid}
        examples.append(example)
    return examples


def generate_query_validator_prompt(query):
    prefix = load_query_validator_prefix()
    validation_examples = get_query_validation_examples()

    validation_example_formatter_template = """
    Input: {input}
    Is valid: {is_valid}\n
    """

    validation_prompt = PromptTemplate(
        input_variables=["input", "is_valid"],
        template=validation_example_formatter_template,
    )

    query_validator_prompt = FewShotPromptTemplate(
        examples=validation_examples,
        example_prompt=validation_prompt,
        prefix=prefix,
        suffix="Input: {input}\nIs valid:",
        input_variables=["input"],
        example_separator="\n",
    )

    return prefix + query_validator_prompt.format(input=query)


def validate_query(query):
    get_moderator().run(query)
    prompt = generate_query_validator_prompt(query)
    res = get_llm().call_as_llm(prompt).strip()
    if res == "N":
        return False
    else:
        return True
