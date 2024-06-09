"""
Aggregator

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

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
    chain = _build_custom_chain(gpt_4o, AGGREGATION_DELEGATION_PATH)
    return chain.invoke({"question": step})


def write_log(log):
    with open("/tmp/log.txt", "a") as f:
        f.write(str(log) + "\n")


### AGGREGATION EXPRESSION ###


def _get_aggregation_constructor(assignee):
    assignee_lower = assignee.lower()
    if "count_values" in assignee_lower:
        return fo.CountValues
    if "count" in assignee_lower:
        return fo.Count
    elif "distinct" in assignee_lower:
        return fo.Distinct
    elif "values" in assignee_lower:
        return fo.Values
    elif "sum" in assignee_lower:
        return fo.Sum
    elif "mean" in assignee_lower:
        return fo.Mean
    elif "max" in assignee_lower:
        return fo.Bounds
    elif "min" in assignee_lower:
        return fo.Bounds
    elif "std" in assignee_lower:
        return fo.Std
    return None


AGGREGATION_EXPRESSION_PATH = os.path.join(
    PROMPTS_DIR, "aggregation_expression.txt"
)


def construct_aggregation(assignee, query, *args, **kwargs):
    aggregation_constructor = _get_aggregation_constructor(assignee)
    if aggregation_constructor is None:
        raise ValueError("Invalid assignee for aggregation")
    elif aggregation_constructor == fo.Count:
        return aggregation_constructor()
    else:
        chain = _build_custom_chain(
            gpt_4o, template_path=AGGREGATION_EXPRESSION_PATH
        )
        expression = chain.invoke({"query": query})
        expression = _format_filter_expression(expression)
        try:
            expression = eval(expression)
            return aggregation_constructor(expression)
        except Exception as e:
            write_log(f"Error: {e}")
            return None


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


def perform_aggregation(query, view, *args, **kwargs):
    write_log("Performing aggregation")
    write_log(f"Query: {query}")
    assignee = delegate_aggregation(query)
    aggregation = construct_aggregation(assignee, query, *args, **kwargs)
    aggregation_results = view.aggregate(aggregation)
    if assignee == "min":
        aggregation_results = aggregation_results[0]
    elif assignee == "max":
        aggregation_results = aggregation_results[1]
    if aggregation_results is None:
        return None, None
    return aggregation, aggregation_results


def stream_aggregation_query(query, view):
    write_log(f"Query: {query}")
    write_log(f"View: {str(view)}")
    aggregation, result = perform_aggregation(query, view)

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


def run_aggregation_query(query, view):
    write_log(f"Query: {query}")
    write_log(f"View: {str(view)}")
    aggregation, result = perform_aggregation(query, view)

    def aggregation_analysis_func(info):
        query = info["query"]
        prompt = _build_aggregation_analysis_prompt(
            query, view, aggregation, result
        )
        response = gpt_4o.invoke(prompt).content
        return {"input": query, "output": response}

    aggregation_analysis_runnable = RunnableLambda(aggregation_analysis_func)
    return aggregation_analysis_runnable.invoke({"query": query})["output"]
