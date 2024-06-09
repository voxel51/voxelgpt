"""
View creation classifier.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

# pylint: disable=relative-beyond-top-level
from .utils import PROMPTS_DIR, _build_custom_chain, gpt_3_5

CREATE_VIEW_CLASSIFICATION_PATH = os.path.join(
    PROMPTS_DIR, "should_create_view_classification.txt"
)


def should_create_view(query):
    chain = _build_custom_chain(gpt_3_5, CREATE_VIEW_CLASSIFICATION_PATH)
    response = chain.invoke({"query": query})
    return "view" in response.lower()
