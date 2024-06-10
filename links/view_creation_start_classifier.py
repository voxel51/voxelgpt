"""
View creation starting point classifier.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

# pylint: disable=relative-beyond-top-level
from .utils import PROMPTS_DIR, _build_custom_chain, gpt_3_5

CREATE_VIEW_STARTING_POINT_CLASSIFICATION_PATH = os.path.join(
    PROMPTS_DIR, "create_view_start_classification.txt"
)


def view_creation_starting_point(query):
    chain = _build_custom_chain(
        gpt_3_5, CREATE_VIEW_STARTING_POINT_CLASSIFICATION_PATH
    )
    response = chain.invoke({"query": query})
    if "add" in response.lower():
        return "view"
    else:
        return "dataset"
