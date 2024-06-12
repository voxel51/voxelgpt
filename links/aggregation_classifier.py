"""
Aggregation classifier.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

# pylint: disable=relative-beyond-top-level
from .utils import PROMPTS_DIR, _build_custom_chain, gpt_3_5

AGGREGATION_CLASSIFICATION_PATH = os.path.join(
    PROMPTS_DIR, "should_aggregate_classification.txt"
)


def should_aggregate(query):
    chain = _build_custom_chain(
        gpt_3_5, template_path=AGGREGATION_CLASSIFICATION_PATH
    )
    response = chain.invoke({"query": query})
    return "yes" in response.lower()
