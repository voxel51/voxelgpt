"""
Tags selector.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

from langchain.prompts import PromptTemplate, FewShotPromptTemplate
import pandas as pd

# pylint: disable=relative-beyond-top-level
from .utils import get_llm, get_cache


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")
PROMPTS_DIR = os.path.join(ROOT_DIR, "prompts")

TAG_SELECTOR_PREFIX_PATH = os.path.join(PROMPTS_DIR, "tag_selector_prefix.txt")
TAG_EXAMPLES_PATH = os.path.join(EXAMPLES_DIR, "tag_selection_examples.csv")


def get_tag_selection_examples():
    cache = get_cache()
    key = "tag_selection_examples"
    if key not in cache:
        df = pd.read_csv(TAG_EXAMPLES_PATH)
        examples = []

        for _, row in df.iterrows():
            example = {
                "candidate_tag": row.candidate_tag,
                "allowed_tags": row.allowed_tags,
                "selected_tags": row.selected_tags,
            }
            examples.append(example)
        cache[key] = examples

    return cache[key]


def load_tag_selector_prefix():
    cache = get_cache()
    key = "tag_prefix"
    if key not in cache:
        with open(TAG_SELECTOR_PREFIX_PATH, "r") as f:
            cache[key] = f.read()
    return cache[key]


def generate_tag_selector_prompt(candidate_tag, allowed_tags):
    prefix = load_tag_selector_prefix()
    tag_selection_examples = get_tag_selection_examples()
    tag_selection_example_formatter_template = """
    Candidate tag: {candidate_tag}
    Allowed tags: {allowed_tags}
    Selected tags: {selected_tags}\n
    """

    tags_prompt = PromptTemplate(
        input_variables=["candidate_tag", "allowed_tags", "selected_tags"],
        template=tag_selection_example_formatter_template,
    )

    tag_selector_prompt = FewShotPromptTemplate(
        examples=tag_selection_examples,
        example_prompt=tags_prompt,
        prefix=prefix,
        suffix="Candidate tag: {candidate_tag}\nAllowed tags: {allowed_tags}\nSelected tags: ",
        input_variables=["candidate_tag", "allowed_tags"],
        example_separator="\n",
    )

    return tag_selector_prompt.format(
        candidate_tag=candidate_tag,
        allowed_tags=", ".join(allowed_tags),
    )


def identify_semantic_matches(candidate_tags, allowed_tags):
    tag_selector_prompt = generate_tag_selector_prompt(
        candidate_tags, allowed_tags
    )
    res = get_llm().call_as_llm(tag_selector_prompt).strip()
    if res[0] == "[" and res[-1] == "]":
        res = res[1:-1]

    semantic_matches = [t.strip().replace("'", "") for t in res.split(",")]
    return [t for t in semantic_matches if t != "" and t in allowed_tags]


def validate_tag(candidate_tag, allowed_tags):
    if candidate_tag in allowed_tags:
        return candidate_tag

    # try matching with case-insensitive
    for tag in allowed_tags:
        if candidate_tag.lower() == tag.lower():
            return tag

    # try matching with prefix
    for tag in allowed_tags:
        if tag.lower().startswith(candidate_tag.lower()):
            return tag
        elif candidate_tag.lower().startswith(tag.lower()):
            return tag

    return None


def select_tags(candidate_tags, allowed_tags):
    selected_tags = []
    for ct in candidate_tags:
        ct_validated = validate_tag(ct, allowed_tags)
        if ct_validated is not None:
            selected_tags.append(ct_validated)
        else:
            sm_tags = identify_semantic_matches(ct, allowed_tags)
            selected_tags.extend(sm_tags)

    return selected_tags
