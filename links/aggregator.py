"""
Aggregator

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import os
from langchain_core.pydantic_v1 import BaseModel, Field

import fiftyone as fo
from fiftyone import ViewField as F

from langchain_core.runnables import RunnableLambda

# pylint: disable=relative-beyond-top-level
from .utils import (
    PROMPTS_DIR,
    _build_custom_chain,
    gpt_4o,
    get_prompt_from,
    _format_filter_expression,
    stream_runnable,
)


AGGREGATION_DELEGATION_PATH = os.path.join(
    PROMPTS_DIR, "aggregation_delegation.txt"
)


def delegate_aggregation(step):
    chain = _build_custom_chain(
        gpt_4o, template_path=AGGREGATION_DELEGATION_PATH
    )
    return chain.invoke({"question": step})


def write_log(log):
    with open("/tmp/log.txt", "a") as f:
        f.write(str(log) + "\n")


### AGGREGATION EXPRESSION ###

AGGREGATION_EXPRESSION_PATH = os.path.join(
    PROMPTS_DIR, "aggregation_expression.txt"
)


class AggregationStage(BaseModel):
    """Aggregation stage"""

    name: str = None

    expression: str = Field(
        description="The expression to apply to the assignee"
    )

    def fiftyone_stage(self):
        raise NotImplementedError

    def apply(self, sample_collection):
        expression = eval(_format_filter_expression(self.expression))
        return sample_collection.aggregate(self.fiftyone_stage()(expression))

    def __repr__(self):
        expression = _format_filter_expression(self.expression)
        return f"{self.name}({expression})"


class Count(AggregationStage):
    """Count aggregation stage"""

    name = "count"

    def fiftyone_stage(self):
        return fo.Count

    def __repr__(self):
        return "count()"

    def apply(self, sample_collection):
        return sample_collection.count()


class CountValues(AggregationStage):
    """CountValues aggregation stage"""

    name = "count_values"

    def fiftyone_stage(self):
        return fo.CountValues


class Distinct(AggregationStage):
    """Distinct aggregation stage"""

    name = "distinct"

    def fiftyone_stage(self):
        return fo.Distinct


class Values(AggregationStage):
    """Values aggregation stage"""

    name = "values"

    def fiftyone_stage(self):
        return fo.Values


class Sum(AggregationStage):
    """Sum aggregation stage"""

    name = "sum"

    def fiftyone_stage(self):
        return fo.Sum


class Mean(AggregationStage):
    """Mean aggregation stage"""

    name = "mean"

    def fiftyone_stage(self):
        return fo.Mean


class Min(AggregationStage):
    """Min aggregation stage"""

    name = "bounds"

    def fiftyone_stage(self):
        return fo.Bounds

    def apply(self, sample_collection):
        expression = eval(_format_filter_expression(self.expression))
        return sample_collection.aggregate(fo.Bounds(expression))[0]

    def __repr__(self):
        expression = _format_filter_expression(self.expression)
        return f"bounds({expression})[0]"


class Max(AggregationStage):
    """Max aggregation stage"""

    name = "bounds"

    def fiftyone_stage(self):
        return fo.Bounds

    def apply(self, sample_collection):
        expression = eval(_format_filter_expression(self.expression))
        return sample_collection.aggregate(fo.Bounds(expression))[1]

    def __repr__(self):
        expression = _format_filter_expression(self.expression)
        return f"bounds({expression})[1]"


class Std(AggregationStage):
    """Std aggregation stage"""

    name = "std"

    def fiftyone_stage(self):
        return fo.Std


def _get_aggregation_constructor(assignee):
    assignee_lower = assignee.lower()
    if "count_values" in assignee_lower:
        return CountValues
    if "count" in assignee_lower:
        return Count
    elif "distinct" in assignee_lower:
        return Distinct
    elif "values" in assignee_lower:
        return Values
    elif "sum" in assignee_lower:
        return Sum
    elif "mean" in assignee_lower:
        return Mean
    elif "max" in assignee_lower:
        return Max
    elif "min" in assignee_lower:
        return Min
    elif "std" in assignee_lower:
        return Std
    return None


def construct_aggregation(assignee, query, view_repr, *args, **kwargs):
    aggregation_constructor = _get_aggregation_constructor(assignee)
    if aggregation_constructor is None:
        raise ValueError("Invalid assignee for aggregation")
    elif aggregation_constructor == Count:
        return aggregation_constructor()
    else:
        prompt = get_prompt_from(AGGREGATION_EXPRESSION_PATH).format(
            aggregation_type=assignee, query=query, view=view_repr
        )
        chain = _build_custom_chain(gpt_4o, prompt=prompt)
        expression = chain.invoke({"query": query})
        write_log(f"Expression: {expression}")

        return aggregation_constructor(expression=expression)


### AGGREGATION ANALYSIS ###

AGGREGATION_ANALYSIS_PATH = os.path.join(
    PROMPTS_DIR, "aggregation_analysis.txt"
)


def _build_aggregation_analysis_prompt(query, view, aggregation, result):
    prompt = get_prompt_from(AGGREGATION_ANALYSIS_PATH).format(
        query=query,
        view_stages=view.view()._stages,
        aggregation=aggregation,
        result=result,
    )
    return prompt


def perform_aggregation(query, view, view_repr, *args, **kwargs):
    write_log("Performing aggregation")
    assignee = delegate_aggregation(query)
    aggregation = construct_aggregation(
        assignee, query, view_repr, *args, **kwargs
    )
    write_log(f"Aggregation: {aggregation}")
    aggregation_results = view.aggregate(aggregation)
    if assignee == "min":
        aggregation_results = aggregation_results[0]
    elif assignee == "max":
        aggregation_results = aggregation_results[1]
    if aggregation_results is None:
        return None, None
    return aggregation, aggregation_results


def stream_aggregation_analysis(query, view, aggregation, result):
    def aggregation_analysis_func_streaming(info):
        query = info["query"]
        prompt = _build_aggregation_analysis_prompt(
            query, view, aggregation, result
        )
        for chunk in gpt_4o.stream(prompt):
            yield chunk

    aggregation_analysis_runnable_streaming = RunnableLambda(
        aggregation_analysis_func_streaming
    )
    for content in stream_runnable(
        aggregation_analysis_runnable_streaming, {"query": query}
    ):
        if isinstance(content, Exception):
            raise content
        yield content.content


def run_aggregation_analysis(query, view, aggregation, result):
    def aggregation_analysis_func(info):
        query = info["query"]
        prompt = _build_aggregation_analysis_prompt(
            query, view, aggregation, result
        )
        response = gpt_4o.invoke(prompt).content
        return {"input": query, "output": response}

    aggregation_analysis_runnable = RunnableLambda(aggregation_analysis_func)
    return aggregation_analysis_runnable.invoke({"query": query})["output"]
