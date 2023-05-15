"""
Generate tests.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(ROOT_DIR, "tests")

TEST_EXAMPLES_PATH = os.path.join(TESTS_DIR, "test_examples.csv")


with open(TEST_EXAMPLES_PATH, "r") as f:
    df = pd.read_csv(f)
    print(df)


prompt = df["query"].tolist()
stages = df["stages"].tolist()
media_type = df["media_type"].tolist()


def prompt_to_func_name(prompt):
    prompt = prompt.replace(" ", "_").replace("?", "").replace(",", "")
    prompt = prompt.replace("(", "").replace(")", "")
    prompt = prompt.replace("[", "").replace("]", "")
    prompt = prompt.replace("‘", "").replace("’", "")
    prompt = prompt.replace("-", "_")
    prompt = prompt.replace(".", "")
    prompt = prompt.replace('"', "").replace("'", "")
    prompt = prompt.replace("“", "").replace("”", "").lower()
    if len(prompt) > 50:
        prompt = prompt[:50]
    return prompt


def format_stages(stages):
    stages = stages.replace('"', "'")
    return stages


def format_prompt(prompt):
    prompt = prompt.replace('"', "'")
    return prompt


def generate_test(prompt, stages, media_type):
    stages = format_stages(stages)

    print(f"def test_{prompt_to_func_name(prompt)}(self):")
    print(f'    prompt = "{format_prompt(prompt)}"')
    if media_type != "video":
        print(f"    " + 'dataset = foz.load_zoo_dataset("quickstart")')
    else:
        print(f"    " + 'dataset = foz.load_zoo_dataset("quickstart-video")')
    print(f'    expected_view = create_view_from_stages("{stages}", dataset)')
    print(f"    " + "gpt_view_stages = get_gpt_view_text(dataset, prompt)")
    print(
        f"    "
        + "gpt_view = create_view_from_stages(gpt_view_stages, dataset)"
    )

    print(f"    assert self.EvaluateResults(expected_view, gpt_view)")
    print("")


for i in range(len(prompt)):
    generate_test(prompt[i], stages[i], media_type[i])
