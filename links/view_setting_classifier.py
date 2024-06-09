"""
View setting classifier.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

# pylint: disable=relative-beyond-top-level
from .utils import PROMPTS_DIR, _build_custom_chain, gpt_3_5

SET_VIEW_CLASSIFICATION_PATH = os.path.join(
    PROMPTS_DIR, "should_set_view_classification.txt"
)


def should_set_view(query):
    chain = _build_custom_chain(gpt_3_5, SET_VIEW_CLASSIFICATION_PATH)
    response = chain.invoke({"query": query})
    return "set" in response.lower()
