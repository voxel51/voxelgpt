"""
Effective Query Generator

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

import fiftyone as fo
from fiftyone import ViewField as F

# pylint: disable=relative-beyond-top-level
from .utils import (
    PROMPTS_DIR,
    _build_custom_chain,
    gpt_4o,
)

EFFECTIVE_QUERY_PATH = os.path.join(
    PROMPTS_DIR, "effective_query_generation.txt"
)


def write_log(log):
    with open("/tmp/log.txt", "a") as f:
        f.write(str(log) + "\n")


def generate_effective_query(chat_history):

    chain = _build_custom_chain(gpt_4o, template_path=EFFECTIVE_QUERY_PATH)
    response = chain.invoke({"chat_history": chat_history})
    write_log(f"Response: {response}")
    return response
