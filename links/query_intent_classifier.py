"""
Query intent classifier.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

# pylint: disable=relative-beyond-top-level
from .utils import PROMPTS_DIR, _build_custom_chain, gpt_3_5

INTENT_CLASSIFICATION_PATH = os.path.join(
    PROMPTS_DIR, "intent_classification.txt"
)

intent_chain = _build_custom_chain(
    gpt_3_5, template_path=INTENT_CLASSIFICATION_PATH
)

allowed_topics = ["documentation", "dataset", "general", "workspace", "other"]

bad_topic_text = "I'm sorry, I'm not sure what you're asking. Could you please provide more context?"


def write_log(log):
    with open("/tmp/log.txt", "a") as f:
        f.write(str(log) + "\n")


def classify_query_intent(query):
    topic = intent_chain.invoke({"query": query}).lower()

    for allowed_topic in allowed_topics:
        if allowed_topic in topic:
            return allowed_topic

    return "other"
