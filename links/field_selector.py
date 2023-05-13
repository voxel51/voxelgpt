"""
Dataset view generator.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")
PROMPTS_DIR = os.path.join(ROOT_DIR, "prompts")

FIELD_SELECTION_EXAMPLES_PATH = os.path.join(
    EXAMPLES_DIR, "fiftyone_field_selection_examples.csv"
)
FIELD_SELECTOR_PREFIX_PATH = os.path.join(
    PROMPTS_DIR, "field_selector_prefix.txt"
)

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")


def get_field_selection_examples():
    df = pd.read_csv(FIELD_SELECTION_EXAMPLES_PATH)
    examples = []

    for _, row in df.iterrows():
        example = {
            "query": row.query,
            "available_fields": row.available_fields,
            "required_fields": row.required_fields,
        }
        examples.append(example)
    return examples


def load_field_selector_prefix():
    with open(FIELD_SELECTOR_PREFIX_PATH, "r") as f:
        return f.read()


def get_field_type(sample, field_name):
    """Assumes dataset has at least one sample"""
    field_type = (
        str(type(sample[field_name]))
        .split(".")[-1]
        .replace("'>", "")
        .replace("<class '", "")
    )
    return field_type


def initialize_available_fields_list():
    id_field = "id: string"
    filepath_field = "filepath: string"
    tags_field = "tags: list"
    return [id_field, filepath_field, tags_field]


def remove_field_from_list(field_names, field_name):
    if field_name in field_names:
        field_names.remove(field_name)
    return field_names


def remove_brain_run_fields(dataset, field_names):
    br_names = dataset.list_brain_runs()
    for brn in br_names:
        br = dataset.get_brain_info(brn)
        if "Similarity" in br.config.cls:
            continue
        elif br.config.method == "hardness":
            field_names = remove_field_from_list(
                field_names, br.config.hardness_field
            )
        elif br.config.method == "mistakenness":
            for name in [
                "mistakenness_field",
                "missing_field",
                "spurious_field",
            ]:
                field_name = getattr(br.config, name)
                field_names = remove_field_from_list(field_names, field_name)
            eval_key = br.config.eval_key
            for field_name in field_names:
                if eval_key in field_name:
                    field_names = remove_field_from_list(
                        field_names, field_name
                    )
        elif br.config.method == "uniqueness":
            field_names = remove_field_from_list(
                field_names, br.config.uniqueness_field
            )
    return field_names


def get_available_fields(dataset):
    sample = dataset.first()
    field_names = list(sample.field_names)

    available_fields = initialize_available_fields_list()

    for fn in remove_brain_run_fields(dataset, field_names):
        if fn in ["id", "filepath", "tags", "metadata"]:
            continue
        field_type = get_field_type(sample, fn)
        available_fields.append(f"{fn}: {field_type}")

    available_fields = "[" + ", ".join(available_fields) + "]"
    return available_fields


def generate_field_selector_prompt(dataset, query):
    available_fields = get_available_fields(dataset)
    prefix = load_field_selector_prefix()
    field_selection_examples = get_field_selection_examples()

    field_selection_example_formatter_template = """
    Query: {query}
    Available fields: {available_fields}
    Required fields: {required_fields}\n
    """

    fields_prompt = PromptTemplate(
        input_variables=["query", "available_fields", "required_fields"],
        template=field_selection_example_formatter_template,
    )

    field_selector_prompt = FewShotPromptTemplate(
        examples=field_selection_examples,
        example_prompt=fields_prompt,
        prefix=prefix,
        suffix="Query: {query}\nAvailable fields: {available_fields}\nRequired fields: ",
        input_variables=["query", "available_fields"],
        example_separator="\n",
    )

    return field_selector_prompt.format(
        query=query, available_fields=available_fields
    )


def format_response(response):
    if response[0] == "[" and response[-1] == "]":
        response = response[1:-1].split(",")
    elif len(response.split(",")) > 1:
        response = response.split(",")
    else:
        response = [response]

    response = [r.strip() for r in response]
    return [r for r in response if r]


def select_fields(dataset, query):
    field_selector_prompt = generate_field_selector_prompt(dataset, query)
    res = llm.call_as_llm(field_selector_prompt).strip()
    return format_response(res)
