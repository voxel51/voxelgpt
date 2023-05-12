"""
View stage description selector.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import json
import os

from langchain.prompts import FewShotPromptTemplate, PromptTemplate


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIEW_STAGES_LIST_PATH = os.path.join(ROOT_DIR, "view_stages_list.txt")
VIEW_STAGE_DESCRIPTIONS_PATH = os.path.join(
    ROOT_DIR, "view_stage_descriptions.json"
)


def get_view_stages_list():
    with open(VIEW_STAGES_LIST_PATH, "r") as f:
        return f.read().splitlines()


def count_view_stage_occurrences(view_stage_examples_prompt):
    view_stages = get_view_stages_list()

    view_stage_counts = {}
    for view_stage in view_stages:
        view_stage_counts[view_stage] = view_stage_examples_prompt.count(
            f"{view_stage}("
        )
    return view_stage_counts


def get_most_relevant_view_stages(view_stage_examples_prompt):
    view_stage_counts = count_view_stage_occurrences(
        view_stage_examples_prompt
    )
    relevant_view_stages = sorted(
        view_stage_counts, key=view_stage_counts.get, reverse=True
    )[:5]
    return relevant_view_stages


def generate_view_stage_descriptions_prompt(view_stage_examples_prompt):
    relevant_view_stages = get_most_relevant_view_stages(
        view_stage_examples_prompt
    )
    with open(VIEW_STAGE_DESCRIPTIONS_PATH, "r") as f:
        view_stage_descriptions_dict = json.load(f)

    examples = [
        {
            "view_stage": view_stage,
            "description": view_stage_descriptions_dict[view_stage][
                "description"
            ],
            "inputs": view_stage_descriptions_dict[view_stage][
                "inputs"
            ].replace("\n", ", "),
        }
        for view_stage in relevant_view_stages
    ]

    example_formatter_template = """
    View stage: {view_stage}
    Description: {description}
    Inputs: {inputs}\n
    """
    example_prompt = PromptTemplate(
        input_variables=["view_stage", "description", "inputs"],
        template=example_formatter_template,
    )

    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Here is some information about view stages you might want to use:\n",
        suffix="",
        input_variables=[],
        example_separator="\n",
    )

    return few_shot_prompt.format()
