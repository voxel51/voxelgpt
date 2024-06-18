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
from .utils import PROMPTS_DIR, _build_chat_chain, gpt_4o, get_prompt_from

CREATE_VIEW_PLANNING_PATH = os.path.join(
    PROMPTS_DIR, "create_view_planning.txt"
)

REVISE_VIEW_PLANNING_PATH = os.path.join(
    PROMPTS_DIR, "revise_view_creation_plan.txt"
)


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

    response = planner.invoke({"messages": [("user", query)]})
    return response


def revise_view_creation_plan(query, inspection_results, view_creation_plan):
    prompt = get_prompt_from(REVISE_VIEW_PLANNING_PATH).format(
        query=query,
        dataset_info=inspection_results,
        initial_plan=view_creation_plan,
    )
    planner = _build_chat_chain(
        gpt_4o,
        prompt=prompt,
        output_type=ViewCreationPlan,
    )
    response = planner.invoke({"messages": [("user", query)]})
    if response is None or response.steps is None:
        return view_creation_plan
    return response
