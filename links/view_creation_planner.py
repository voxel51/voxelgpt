"""
View creation classifier.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

# pylint: disable=relative-beyond-top-level
from .utils import PROMPTS_DIR, _build_chat_chain, gpt_4o

CREATE_VIEW_PLANNING_PATH = os.path.join(
    PROMPTS_DIR, "create_view_planning.txt"
)


def write_log(log):
    with open("log.txt", "w") as f:
        f.write(str(log))


class ViewCreationPlan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


def create_view_creation_plan(query):
    planner = _build_chat_chain(
        gpt_4o,
        template_path=CREATE_VIEW_PLANNING_PATH,
        output_type=ViewCreationPlan,
    )
    write_log("inside create_view_creation_plan")
    write_log("created planner")

    response = planner.invoke({"messages": [("user", query)]})
    write_log("inside create_view_creation_plan")
    write_log(str(response))

    return planner.invoke({"messages": [("user", query)]})
