"""
VoxelGPT entrypoints.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import re
import sys

import fiftyone as fo

from links.query_intent_classifier import classify_query_intent
from links.docs_query_dispatcher import run_docs_query, stream_docs_query
from links.computer_vision_query_dispatcher import (
    run_computer_vision_query,
    stream_computer_vision_query,
)
from links.workspace_inspection_agent import run_workspace_inspection_query
from links.data_inspection_agent import run_basic_data_inspection_query
from links.view_creation_classifier import should_create_view
from links.view_setting_classifier import should_set_view
from links.aggregation_classifier import should_aggregate
from links.view_creator import create_view
from links.aggregator import (
    run_aggregation_query,
    stream_aggregation_query,
)

# from links.effective_query_generator import generate_effective_query


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

    def _respond(message, overwrite=False):
        if isinstance(message, str):
            message = {"string": message, "markdown": message}

        str_msg = message.get("string", None)
        if str_msg is not None:
            _log_chat_history("GPT", str_msg, chat_history)

        if dialect == "raw":
            return str_msg

        msg = message.get(dialect, None)
        if msg is not None:
            return _emit_message(msg, str_msg, overwrite=overwrite)

    if ctx is not None:
        sample_collection = ctx.view

    view = sample_collection.view()

    # def perform_docs_query(query):
    #     write_log("in perform_docs_query")
    #     if allow_streaming:
    #         write_log("in if")
    #         message = ""
    #         for content in stream_docs_query(query):
    #             if isinstance(content, dict):
    #                 message = content
    #             else:
    #                 message += content
    #                 yield _emit_streaming_content(content)

    #         yield _emit_streaming_content("", last=True)
    #         yield _respond(_format_docs_message(message), overwrite=True)
    #     else:
    #         write_log("in else")
    #         yield _respond(_format_docs_message(run_docs_query(query)))

    # def perform_cv_query(query):
    #     if allow_streaming:
    #         message = ""
    #         for content in stream_computer_vision_query(query):
    #             message += content
    #             yield _emit_streaming_content(content)

    #         yield _emit_streaming_content("", last=True)
    #         yield _respond(message, overwrite=True)
    #     else:
    #         yield _respond(run_computer_vision_query(query))

    if query.strip().lower() == "help":
        yield _respond(_help_message())
        return

    _log_chat_history("User", query, chat_history)

    # Generate a new query that incorporates the chat history
    # if chat_history:
    #     query = generate_effective_query(chat_history)

    # Intent classification
    intent = classify_query_intent(query)
    write_log(intent)

    if intent == "documentation":
        write_log("performing docs query")
        if allow_streaming:
            write_log("in if")
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
            write_log("in else")
            yield _respond(_format_docs_message(run_docs_query(query)))
        return
    elif intent == "general":
        write_log("performing cv query")
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

    ## If the query is not a documentation query, a computer vision query, or a workspace query, we need to inspect the dataset
    if sample_collection is None:
        yield _respond(
            "You must provide a sample collection in order for me to respond "
            "to this query"
        )
        return

    ## If no view creation and no aggregation, run basic data inspection agent
    create_view_flag = should_create_view(query)
    aggregate_flag = should_aggregate(query)

    if not create_view_flag and not aggregate_flag:
        yield _respond(
            run_basic_data_inspection_query(query, sample_collection)
        )
        # return

    if create_view_flag:
        view = create_view(query, sample_collection)

    write_log(view)

    if should_set_view(query):
        yield _emit_view(view.view())

    if not aggregate_flag:
        return

    write_log("Aggregating")
    if allow_streaming:
        write_log("in if")
        message = ""
        for content in stream_aggregation_query(query, view):
            message += content
            yield _emit_streaming_content(content)

        yield _emit_streaming_content("", last=True)
        yield _respond(message, overwrite=True)
    else:
        write_log("in else")
        yield _respond(run_aggregation_query(query, view))

    return

    # if "metadata" in ".".join(stages) and "metadata" not in runs:
    #     stages = "_NEED_METADATA_"

    # if dialect == "raw":
    #     yield stages
    #     return

    # if stages == "_NEED_METADATA_":
    #     yield _respond(_metadata_message())
    #     return

    # if stages == "_MORE_":
    #     yield _respond(_specific_message())
    #     return

    # if stages == "_CONFUSED_":
    #     yield _respond(_clarify_message())
    #     return


def _format_label_classes(label_classes):
    unmatched_entities = []

    label_fields = list(label_classes.keys())
    label_class_dict = {}
    for field in label_fields:
        field_list = []
        lcl = label_classes[field]
        for el in lcl:
            el_val = list(el.values())[0]
            if type(el_val) == str:
                field_list.append(el_val)
            else:
                unmatched_entities.append(list(el.keys())[0])
                field_list += el_val

        label_class_dict[field] = field_list

    return label_class_dict, unmatched_entities


def _reformat_query(examples, label_classes):
    example_lines = examples.split("\n")
    query = example_lines[-2]

    label_classes_list = list(label_classes.values())
    label_classes_list = [
        item for sublist in label_classes_list for item in sublist
    ]
    class_name_map = {k: v for d in label_classes_list for k, v in d.items()}
    for k, v in class_name_map.items():
        if type(v) == str:
            query = query.replace(k, v)
        elif v:
            clarification = f" where by {k} I mean any of {v}"
            query += clarification

    example_lines[-2] = query
    return "\n".join(example_lines)


def _log_chat_history(speaker, text, chat_history):
    chat_history.append(f"{speaker}: {text}")


def _metadata_message():
    return {
        "string": "Please compute metadata and then try again",
        "markdown": (
            "Please run `compute_metadata()` on your samples and then try "
            "again"
        ),
    }


def _clarify_message():
    return "I'm sorry, I don't understand. Can you clarify what you're asking?"


def _specific_message():
    return "I'm sorry, I don't understand. Can you be more specific?"


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


def _algorithms_message(algorithms):
    prefix = "Identified potential algorithms: "
    markdown = ", ".join(f"`{a}`" for a in algorithms)
    return {
        "string": prefix + ", ".join(algorithms),
        "markdown": prefix + markdown,
    }


def _runs_message(runs, runs_message):
    prefix = "Identified potential runs: "
    runs_map = {}
    for run_type in list(runs.keys()):
        run_info = runs[run_type]
        if run_type == "uniqueness":
            key = "uniqueness_field"
        else:
            key = "key"

        runs_map[run_type] = run_info[key]

    # String
    str_message = ""
    if runs_map:
        str_message += prefix + str(runs_map)

    if runs_message:
        str_message += "\n" + runs_message + "\n"

    # Markdown
    runs_md = ""
    if runs_map:
        chunks = []
        for run_type, key in runs_map.items():
            chunks.append(f"\n - `{run_type}` run: `{key}`")

        runs_md += prefix + "".join(chunks)

    if runs_message:
        runs_md += "\n" + runs_message + "\n"

    return {
        "string": str_message,
        "markdown": runs_md,
    }


def _fields_message(fields):
    prefix = "Identified potential fields: "
    markdown = ", ".join(f"`{f}`" for f in fields)
    return {
        "string": prefix + ", ".join(fields),
        "markdown": prefix + markdown,
    }


def _label_classes_message(label_classes):
    prefix = "Identified potential label classes: "

    # Markdown
    chunks = []
    for label_field, classes in label_classes.items():
        chunks.append(f"\n - `{label_field}` field: ")
        if classes:
            chunks.append(", ".join(f"`{c}`" for c in sorted(classes)))
        else:
            chunks.append("N/A")

    markdown = "".join(chunks)

    return {
        "string": prefix + str(label_classes),
        "markdown": prefix + markdown,
    }


_HELP_MESSAGE_MARKDOWN = """
Hi! I'm VoxelGPT, your AI assistant for computer vision.

I can help you with the following tasks:
- 🔎 **Querying your data:** I can help you filter, match, sort, and more - without writing a line of code. Tell me what you'd like to see and I'll load the corresponding view
- 💪 **Becoming a FiftyOne pro:** I have access to the FiftyOne documentation, so I can help you learn how to use FiftyOne and find the information you're looking for
- 📈 **Troubleshooting data quality:** I can help you build better datasets and higher quality models by answering general knowledge questions about computer vision and machine learning

**Tips**
- Be as specific as possible. The more specific you are, the better I can help you. I am still learning, so sometimes I need a little help understanding what you're asking
- If you want to query your dataset, but your input is being interpreted as a documentation or general computer vision query, try using the `show` keyword. For example: *"show me all images with a label of dog"*
- If you want to query the FiftyOne documentation, try using either `docs` or `fiftyone` in your query. For example: *"how do I load a dataset in FiftyOne?"*
- If you want me to use our conversation history to infer what you're asking, try using the `now` keyword. For example: if you just asked *"show me high confidence predictions of cats, dogs, and rabbits"*, you can ask *"now the low confidence predictions"*

**Learn more**
- You can learn more about me on my [GitHub page](https://github.com/voxel51/voxelgpt). While you're at it, please give me a star ⭐! VoxelGPT is open source and it is constantly improving. Contributions are welcome!
- Did you know that I'm a [FiftyOne Plugin](https://docs.voxel51.com/plugins/index.html)? Check out how FiftyOne can be extended to do all sorts of cool things!
- Learn more about [FiftyOne](https://github.com/voxel51/fiftyone) and give the project a star ⭐! FiftyOne is open source too!
- Join the [FiftyOne Slack community](https://slack.voxel51.com) where thousands of enthusiasts and professionals are discussing the latest in computer vision and machine learning

I'm still learning, so I appreciate your patience 😊
"""

_HELP_MESSAGE_STRING = """
Hi! I'm VoxelGPT, your AI assistant for computer vision.

I can help you with the following tasks
===============================================================================

🔎  ~~Querying your data~~
    I can help you filter, match, sort, and more - without writing a line of
    code. Tell me what you'd like to see and I'll load the corresponding view

💪  ~~Becoming a FiftyOne pro~~
    I have access to the FiftyOne documentation, so I can help you learn how to
    use FiftyOne and find the information you're looking for

📈  ~~Troubleshooting data quality~~
    I can help you build better datasets and higher quality models by answering
    general knowledge questions about computer vision and machine learning

Tips
===============================================================================

1.  ~~Be as specific as possible~~
    The more specific you are, the better I can help you. I am still learning,
    so sometimes I need a little help understanding what you're asking

2.  If you want to query your dataset, but your input is being interpreted as a
    documentation or general computer vision query, try using the 'show'
    keyword. For example:

        show me all images with a label of dog

3.  If you want to query the FiftyOne documentation, try using either 'docs' or
   'fiftyone' in your query. For example:

        how do I load a dataset in FiftyOne?

4.  If you want me to use our conversation history to infer what you're asking,
    try using the 'now' keyword. For example:

        show me high confidence predictions of cats, dogs, and rabbits
        now the low confidence predictions

5.  Type 'reset' to clear our chat history

6.  Type 'exit' to exit our chat

Learn more
===============================================================================

-   You can learn more about me on GitHub: https://github.com/voxel51/voxelgpt
    While you're at it, please give me a star ⭐! VoxelGPT is an open source
    project and it is constantly improving. Contributions are welcome!

-   Did you know that I'm a FiftyOne Plugin? Check out how FiftyOne can be 
    extended to do all sorts of cool things at https://docs.voxel51.com/plugins/index.html

-   Learn more about FiftyOne at https://github.com/voxel51/fiftyone
    Please give the project a star ⭐! FiftyOne is open source too!

-   Join the FiftyOne Slack community at https://slack.voxel51.com
    Thousands of enthusiasts and professionals are discussing the latest in
    computer vision and machine learning

I'm still learning, so I appreciate your patience 😊
"""


def _help_message():
    return {
        "string": _HELP_MESSAGE_STRING.strip(),
        "markdown": _HELP_MESSAGE_MARKDOWN.strip(),
    }


def _view_stages_message(view_stages):
    prefix = "Identified potential view stages: "
    markdown = ", ".join(_wrap_stage_name(name) for name in view_stages)
    return {
        "string": prefix + str(view_stages),
        "markdown": prefix + markdown,
    }


def _wrap_stage_name(stage_name):
    return f"[{stage_name}()]({_get_stage_doc_link(stage_name)})"


def _get_stage_doc_link(stage_name):
    return f"https://docs.voxel51.com/api/fiftyone.core.collections.html#fiftyone.core.collections.SampleCollection.{stage_name}"


def _load_view_message(stages):
    prefix = "Okay, I'm going to load "
    view_str = ".".join(stages)

    # Markdown
    if len(view_str) < 80 or len(stages) <= 2:
        markdown = f":\n```py\n{view_str}\n```"
    else:
        stages_str = "".join(f"    .{s}\n" for s in stages[1:])
        markdown = f":\n```py\nview = (\n    {stages[0]}\n{stages_str})\n```"

    return {
        "string": prefix + view_str,
        "markdown": prefix + markdown,
    }


def _invalid_view_message():
    return "I tested the view and it was invalid. Please try again"


def _full_collection_message(sample_collection):
    if isinstance(sample_collection, fo.DatasetView):
        ctype = "view"
    else:
        ctype = "dataset"

    return f"Okay, let's load your entire {ctype}"


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


def _emit_warning(message):
    return {"type": "warning", "data": {"message": message}}
