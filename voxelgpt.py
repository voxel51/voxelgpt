"""
VoxelGPT entrypoints.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import os
import re
import sys

import fiftyone as fo

from links.query_intent_classifier import classify_query_intent
from links.docs_qa_with_sources import run_docs_query, stream_docs_query
from links.general_qa import (
    run_computer_vision_query,
    stream_computer_vision_query,
)
from links.workspace_inspection import run_workspace_inspection_query
from links.data_inspection import (
    run_basic_data_inspection_query,
    _run_default_inspection_for_plan,
)
from links.view_creation_classifier import (
    should_create_view,
    should_add_to_view,
)
from links.view_setting_classifier import should_set_view
from links.aggregation_classifier import should_aggregate

from links.view_creator import create_view_from_plan
from links.aggregator import (
    delegate_aggregation,
    construct_aggregation,
    stream_aggregation_analysis,
    run_aggregation_analysis,
)
from links.view_creation_planner import (
    create_view_creation_plan,
    revise_view_creation_plan,
)
from links.view_stage_delegator import delegate_view_stage_creation
from links.utils import PROMPTS_DIR, get_prompt_from

from links.effective_query_generator import generate_effective_query


_SUPPORTED_DIALECTS = ("string", "markdown", "raw")


## For debugging purposes
def write_log(log):
    with open("/tmp/log.txt", "a") as f:
        f.write(str(log) + "\n")


def ask_voxelgpt_interactive(
    sample_collection=None,
    session=None,
    chat_history=None,
):
    """Launches an interactive session with VoxelGPT.

    You will be prompted by ``input()`` to provide queries, any responses from
    VoxelGPT will be printed to stdout, and any views created are automatically
    loaded in the App.

    If you provide a chat history, your query and VoxelGPT's responses will be
    added to it.

    Special keywords:

    -   Type `help` to see a help message
    -   Type `reset` to clear your chat history
    -   Type `exit` or `^c` to end your session

    Args:
        sample_collection (None): a
            :class:`fiftyone.core.collections.SampleCollection` to query
        session (None): an optional :class:`fiftyone.core.session.Session` to
            load views in. By default, a new App session is launched
        chat_history (None): an optional chat history list
    """
    if chat_history is None:
        chat_history = []

    empty = 0

    while True:
        if empty >= 5:
            query = input("How can I help you? (try 'help' or 'exit') ")
        else:
            query = input("How can I help you? ")

        if not query:
            empty += 1
            continue

        if query.strip().lower() == "exit":
            break

        if query.strip().lower() == "reset":
            chat_history.clear()
            continue

        empty = 0

        coll = ask_voxelgpt(
            query,
            sample_collection=sample_collection,
            chat_history=chat_history,
        )

        if coll is None:
            continue

        if session is None:
            session = fo.launch_app(sample_collection, auto=False)

        if session._collection != coll:
            if isinstance(coll, fo.Dataset):
                session.dataset = coll
            elif isinstance(coll, fo.DatasetView):
                session.view = coll


def ask_voxelgpt(
    query,
    sample_collection=None,
    ctx=None,
    allow_streaming=True,
    chat_history=None,
):
    """Prompts VoxelGPT with the given query with respect to the given sample
    collection.

    If your query is understood as a view to load, it will be returned.

    If you provide a chat history, your query and VoxelGPT's responses will be
    added to it.

    Args:
        query: a prompt string
        sample_collection (None): a
            :class:`fiftyone.core.collections.SampleCollection` to query
        allow_streaming (True): whether to allow streaming responses
        chat_history (None): an optional chat history list

    Returns:
        a :class:`fiftyone.core.view.DatasetView`, or None if the query did not
        result in a view creation
    """
    view = None

    for response in ask_voxelgpt_generator(
        query,
        sample_collection=sample_collection,
        ctx=ctx,
        dialect="string",
        allow_streaming=allow_streaming,
        chat_history=chat_history,
    ):
        type = response["type"]
        data = response["data"]

        if type == "view":
            view = data["view"]
        elif type == "message":
            if not data["overwrite"]:
                print(data["message"])
        elif type == "streaming":
            sys.stdout.write(data["content"])
            if data["last"]:
                sys.stdout.write("\n")

            sys.stdout.flush()

    return view


def _view_creation_plan_message(plan):
    message = ""
    for step in plan.steps:
        message += f"  - {step}\n"
    return {
        "string": f"Here's the plan:\n{message}",
        "markdown": f"Here's the plan:\n{message}",
    }


def _get_dataset_and_view(sample_collection, ctx):
    if sample_collection is not None:
        if isinstance(sample_collection, fo.DatasetView):
            view = sample_collection
            dataset = sample_collection._dataset
        else:
            view = None
            dataset = sample_collection
    elif ctx is not None:
        view = ctx.view
        dataset = ctx.dataset
    else:
        view = None
        dataset = None

    return dataset, view


def ask_voxelgpt_generator(
    query,
    sample_collection=None,
    ctx=None,
    dialect="string",
    allow_streaming=True,
    chat_history=None,
):
    """Generator that emits responses from VoxelGPT with respect to the given
    query.

    The generator may emit the following types of content:

    -   Messages in the format::

        {
            "type": "message",
            "data": {
                "message": message,         # in your chosen dialect
                "history": history,         # string added to `chat_history`
                "overwrite": True/False     # overwrite previous message?
            }
        }

    -   Streaming content in the format:

        {
            "type": "streaming",
            "data": {
                "content": content,         # a chunk of streaming content
                "last": True/False          # last chunk in the stream?
            }
        }

    -   Views in the format::

        {
            "type": "view",
            "data": {
                "view": view
            }
        }

    -   Warnings in the formatt::

        {
            "type": "warning",
            "data": {
                "message": message
            }
        }

    You can use the ``dialect`` parameter to configure the message format.

    If you provide a chat history, your query and VoxelGPT's responses will be
    added to it.

    Args:
        query: a prompt string
        sample_collection (None): a
            :class:`fiftyone.core.collections.SampleCollection` to query
        dialect ("string"): the response format to return. Supported values are
            ``("string", "markdown", "raw")``
        allow_streaming (True): whether to allow streaming responses
        chat_history (None): an optional chat history list
    """
    if dialect not in _SUPPORTED_DIALECTS:
        raise ValueError(
            f"Unsupported dialect '{dialect}'. Supported: {_SUPPORTED_DIALECTS}"
        )

    if chat_history is None:
        chat_history = []

    def _respond(message, overwrite=False, add_to_history=True):
        if isinstance(message, str):
            message = {"string": message, "markdown": message}

        str_msg = message.get("string", None)
        if str_msg is not None and add_to_history:
            _log_chat_history("VoxelGPT", str_msg, chat_history)

        if dialect == "raw":
            return str_msg

        msg = message.get(dialect, None)
        if msg is not None:
            return _emit_message(msg, str_msg, overwrite=overwrite)

    dataset, current_view = _get_dataset_and_view(sample_collection, ctx)
    view_message = None

    if query.strip().lower() == "help":
        yield _respond(_help_message())
        return

    _log_chat_history("User", query, chat_history)

    # Generate a new query that incorporates the chat history
    if chat_history:
        query = generate_effective_query(chat_history)

    # Intent classification
    intent = classify_query_intent(query)

    if intent == "documentation":
        if allow_streaming:
            message = ""
            for content in stream_docs_query(query):
                if isinstance(content, dict):
                    message = content
                else:
                    message += content
                    yield _emit_streaming_content(content)

            yield _emit_streaming_content("", last=True)
            yield _respond(_format_docs_message(message), overwrite=True)
        else:
            yield _respond(_format_docs_message(run_docs_query(query)))
        return
    elif intent == "general":
        if allow_streaming:
            message = ""
            for content in stream_computer_vision_query(query):
                message += content
                yield _emit_streaming_content(content)

            yield _emit_streaming_content("", last=True)
            yield _respond(message, overwrite=True)
        else:
            yield _respond(run_computer_vision_query(query))
        return
    elif intent == "workspace":
        yield _respond(
            _format_docs_message(run_workspace_inspection_query(query))
        )
        return
    elif intent == "other":
        yield _respond(_clarify_message())
        return

    if dataset is None:
        yield _respond(
            "You must provide a sample collection in order for me to respond "
            "to this query"
        )
        return

    create_view_flag = should_create_view(query)
    aggregate_flag = should_aggregate(query)

    ## If no view creation and no aggregation, run basic data inspection agent
    if not create_view_flag and not aggregate_flag:
        query_view = current_view if current_view is not None else dataset
        yield _respond(run_basic_data_inspection_query(query, query_view))
        return

    ### VIEW CREATION
    if current_view is not None and should_add_to_view(query, current_view):
        starting_view = current_view
        starting_str = "view"
    else:
        starting_view = dataset
        starting_str = "dataset"

    yield _respond("Creating a plan...", add_to_history=False)
    view_creation_plan = create_view_creation_plan(query)
    yield _respond(
        _view_creation_plan_message(view_creation_plan), add_to_history=False
    )
    view_creation_actors = [
        delegate_view_stage_creation(step) for step in view_creation_plan.steps
    ]
    yield _respond("Inspecting the data schema...", add_to_history=False)
    inspection_results = _run_default_inspection_for_plan(
        starting_view, view_creation_actors, view_creation_plan
    )
    yield _respond("Crafting a revised plan...", add_to_history=False)
    view_creation_plan = revise_view_creation_plan(
        query, inspection_results, view_creation_plan
    )
    yield _respond(
        _view_creation_plan_message(view_creation_plan), add_to_history=False
    )

    view, stage_reprs = create_view_from_plan(
        starting_view, view_creation_plan
    )

    if view is None:
        yield _respond(_invalid_view_message())
        return

    if view == starting_view:
        yield _respond(
            "No view stages were applied. Perhaps you should try a different query, or add fields to the dataset.",
        )
    view_message = _load_view_message(starting_str, stage_reprs)
    yield _respond(view_message)

    if should_set_view(query):
        yield _emit_view(view.view())

    ### AGGREGATION ###
    if not aggregate_flag:
        yield _respond(
            "I've updated the view in the App. Let me know if you need anything else!",
            add_to_history=False,
        )
        return

    if aggregate_flag:
        aggregation_assignee = delegate_aggregation(query)

        aggregation = construct_aggregation(
            aggregation_assignee, query, view_message["markdown"]
        )
        if aggregation is None:
            yield _respond(
                "I'm sorry, I couldn't understand the aggregation query"
            )
            return

        if create_view_flag:
            yield _respond(
                _perform_aggregation_message(
                    "view", str(aggregation.__repr__())
                )
            )
        else:
            yield _respond(
                _perform_aggregation_message(
                    "dataset", str(aggregation.__repr__())
                )
            )

        aggregation_results = aggregation.apply(view)
        if aggregation_results is None:
            yield _respond("I'm sorry, I couldn't perform the aggregation")
            return

        if allow_streaming:
            message = ""
            for content in stream_aggregation_analysis(
                query, view, aggregation, aggregation_results
            ):
                message += content
                yield _emit_streaming_content(content)

            yield _emit_streaming_content("", last=True)
            yield _respond(message, overwrite=True)
        else:
            yield _respond(
                run_aggregation_analysis(
                    query, view, aggregation, aggregation_results
                )
            )

    return


def _log_chat_history(speaker, text, chat_history):
    chat_history.append(f"{speaker}: {text}")


def _format_docs_message(response):
    # Markdown
    # Convert all URLs to [url](url)
    patt = r"(https?://[^\s]+)"
    repl = r"[\1](\1)"
    md_response = re.sub(patt, repl, response)

    return {
        "string": response,
        "markdown": md_response,
    }


def _help_message():
    return {
        "string": _HELP_MESSAGE_STRING.strip(),
        "markdown": _HELP_MESSAGE_MARKDOWN.strip(),
    }


def _perform_aggregation_message(start_string, aggregation_string):
    return {
        "string": f"Performing aggregation: {start_string}.{aggregation_string}",
        "markdown": f"Performing aggregation:\n```py\n{start_string}.{aggregation_string}\n```",
    }


def _load_view_message(start_string, view_stage_strings):
    if not view_stage_strings:
        return {
            "string": "Not applying any view stages.",
            "markdown": "Not applying any view stages.",
        }
    prefix = "Okay, I'm going to load "
    view_str = start_string + "." + ".".join(view_stage_strings)

    # Markdown
    if len(view_str) < 80 or len(view_stage_strings) <= 2:
        markdown = f":\n```py\n{view_str}\n```"
    else:
        stages_str = "".join(f"    .{s}\n" for s in view_stage_strings[1:])
        markdown = f":\n```py\nview = (\n    {view_stage_strings[0]}\n{stages_str})\n```"

    return {
        "string": prefix + "`" + view_str + "`",
        "markdown": prefix + markdown,
    }


def _invalid_view_message():
    return "I tested the view and it was invalid. Please try again"


def _clarify_message():
    return "I'm sorry, I don't understand. Can you clarify what you're asking?"


def _emit_message(message, history, overwrite=False):
    return {
        "type": "message",
        "data": {
            "message": message,
            "history": history,
            "overwrite": overwrite,
        },
    }


def _emit_streaming_content(content, last=False):
    return {"type": "streaming", "data": {"content": content, "last": last}}


def _emit_view(view):
    return {"type": "view", "data": {"view": view}}


HELP_MESSAGE_MD_PATH = os.path.join(PROMPTS_DIR, "help_message_markdown.txt")
_HELP_MESSAGE_MARKDOWN = get_prompt_from(HELP_MESSAGE_MD_PATH)

HELP_MESSAGE_STRING_PATH = os.path.join(PROMPTS_DIR, "help_message_string.txt")
_HELP_MESSAGE_STRING = get_prompt_from(HELP_MESSAGE_STRING_PATH)
