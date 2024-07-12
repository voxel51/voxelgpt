"""
View creation classifier.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import os

# pylint: disable=relative-beyond-top-level
from .utils import (
    PROMPTS_DIR,
    _build_custom_chain,
    gpt_3_5,
    gpt_4o,
    protect_text,
)

CREATE_VIEW_CLASSIFICATION_PATH = os.path.join(
    PROMPTS_DIR, "should_create_view_classification.txt"
)


def should_create_view(query):
    chain = _build_custom_chain(
        gpt_3_5, template_path=CREATE_VIEW_CLASSIFICATION_PATH
    )
    response = chain.invoke({"query": query})
    return "view" in response.lower()


ADD_TO_VIEW_CLASSIFICATION_PATH = os.path.join(
    PROMPTS_DIR, "add_to_existing_view_classification.txt"
)


def _format(view):
    stages = view.view()._stages
    if len(stages) == 0:
        return "dataset"
    view_str = ""
    for stage in stages:
        view_str += f"   - {protect_text(str(stage))}\n"
    return view_str


_view_words = ("view", "add", "now")


def should_add_to_view(query, view, view_kw_flag=None, dataset_kw_flag=None):
    if view_kw_flag or any(word in query.lower() for word in _view_words):
        return True
    if dataset_kw_flag or "dataset" in query.lower():
        return False

    chain = _build_custom_chain(
        gpt_4o, template_path=ADD_TO_VIEW_CLASSIFICATION_PATH
    )
    response = chain.invoke({"query": query, "current_view": _format(view)})
    return "add" in response.lower()
