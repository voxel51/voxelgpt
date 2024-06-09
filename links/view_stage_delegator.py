"""
View creation classifier.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

# pylint: disable=relative-beyond-top-level
from .utils import PROMPTS_DIR, _build_custom_chain, gpt_4o


VIEW_STAGE_DELEGATION_PATH = os.path.join(
    PROMPTS_DIR, "create_view_stage_delegation.txt"
)


def delegate_view_stage_creation(step):
    chain = _build_custom_chain(gpt_4o, VIEW_STAGE_DELEGATION_PATH)
    return chain.invoke({"question": step})
