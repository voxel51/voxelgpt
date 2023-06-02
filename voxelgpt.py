"""
VoxelGPT entrypoints.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from collections import defaultdict
import re
import sys

import fiftyone as fo

from links.query_moderator import moderate_query
from links.query_intent_classifier import classify_query_intent
from links.docs_query_dispatcher import run_docs_query, stream_docs_query
from links.computer_vision_query_dispatcher import (
    run_computer_vision_query,
    stream_computer_vision_query,
)
from links.view_stage_example_selector import (
    generate_view_stage_examples_prompt,
)
from links.view_stage_description_selector import (
    generate_view_stage_descriptions_prompt,
    get_most_relevant_view_stages,
)
from links.algorithm_selector import select_algorithms
from links.run_selector import select_runs
from links.field_selector import select_fields
from links.label_class_selector import select_label_classes
from links.dataset_view_generator import get_gpt_view_stage_strings
from links.effective_query_generator import generate_effective_query


_SUPPORTED_DIALECTS = ("string", "markdown", "raw")


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

    .. note::

        Type `exit` or `^c` to end your session, or type `reset` to clear your
        chat history.

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
            query = input("How can I help you? ('exit' to quit) ")
        else:
            query = input("How can I help you? ")

        if not query:
            empty += 1
            continue

        if query.lower() == "exit":
            break

        if query.lower() == "reset":
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

    if not moderate_query(query):
        yield _respond(_moderation_fail_message())
        return

    _log_chat_history("User", query, chat_history)

    if "help" in query.lower():
        yield _respond(_help_message())
        return

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
    elif intent == "computer_vision":
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
    elif intent != "display":
        yield _respond(_clarify_message())
        return

    if sample_collection is None:
        yield _respond(
            "You must provide a sample collection in order for me to respond "
            "to this query"
        )
        return

    if sample_collection.media_type not in ("image", "video"):
        yield _respond(
            "Only image or video collections are currently supported"
        )
        return

    # Algorithms
    algorithms = select_algorithms(query)
    if algorithms:
        yield _respond(_algorithms_message(algorithms))

    # Runs
    runs, runs_message = select_runs(sample_collection, query, algorithms)
    if runs or runs_message:
        yield _respond(_runs_message(runs, runs_message))

    # Fields
    fields = select_fields(sample_collection, query)
    if fields:
        yield _respond(_fields_message(fields))

    # Label classes
    label_classes = select_label_classes(sample_collection, query, fields)
    if label_classes == "_CONFUSED_":
        if "text_similarity" in runs:
            label_classes = {}
        else:
            yield _respond(_clarify_message())
            return

    if any(len(v) > 0 for v in label_classes.values()):
        _label_classes, _unmatched_classes = _format_label_classes(
            label_classes
        )
        yield _respond(_label_classes_message(_label_classes))

    examples = generate_view_stage_examples_prompt(
        sample_collection, query, runs, label_classes
    )
    view_stage_descriptions = generate_view_stage_descriptions_prompt(examples)

    # View stages
    view_stages = get_most_relevant_view_stages(examples)
    if view_stages:
        yield _respond(_view_stages_message(view_stages))

    examples = _reformat_query(examples, label_classes)
    _label_classes, _unmatched_classes = _format_label_classes(label_classes)

    stages = get_gpt_view_stage_strings(
        sample_collection,
        runs,
        fields,
        _label_classes,
        _unmatched_classes,
        view_stage_descriptions,
        examples,
    )

    if "metadata" in ".".join(stages) and "metadata" not in runs:
        stages = "_NEED_METADATA_"

    if dialect == "raw":
        yield stages
        return

    if stages == "_NEED_METADATA_":
        yield _respond(_metadata_message())
        return

    if stages == "_MORE_":
        yield _respond(_specific_message())
        return

    if stages == "_CONFUSED_":
        yield _respond(_clarify_message())
        return

    stages = _format_stages(sample_collection, stages)

    if stages:
        yield _respond(_load_view_message(stages))

        try:
            view = _build_view(sample_collection, stages)
            yield _emit_view(view)
        except Exception as e:
            yield _respond(_invalid_view_message())
    else:
        yield _respond(_full_collection_message(sample_collection))
        yield _emit_view(sample_collection)


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


def _format_stages(sample_collection, stages):
    stages = [s for s in stages if s.strip() not in ("", "None")]

    if not stages:
        return None

    if isinstance(sample_collection, fo.DatasetView):
        ctype = "view"
    else:
        ctype = "dataset"

    return [ctype] + stages


def _build_view(sample_collection, stages):
    import fiftyone as fo
    from fiftyone import ViewField as F

    if isinstance(sample_collection, fo.DatasetView):
        view = sample_collection
        dataset = view._root_dataset
    else:
        dataset = sample_collection
        view = dataset.view()

    view = eval(".".join(stages))

    # Ensures view is valid
    _ = view.count()

    return view


def _log_chat_history(speaker, text, chat_history):
    chat_history.append(f"{speaker}: {text}")


def _moderation_fail_message():
    return "I'm sorry, this query does not abide by OpenAI's guidelines"


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


def _help_message():
    md_message = """
    Hi there! I'm VoxelGPT, your AI assistant for computer vision. Because your 
    query included `help`, I'm going to share some information to help you get 
    started. 
    
    I can help you with the following tasks:
    - **Query Your Dataset**: I can help you filter, match, sort, and more - all without writing a single line of code!
    - **Become a FiftyOne Pro**: I can help you learn how to use FiftyOne. I have access to the FiftyOne documentation, and I can help you find what you're looking for.
    - **Troubleshoot Data Quality Issues**: I can help you build better datasets and higher quality models.


    **Here are a few tips:**
    - *Be as specific as possible*. The more specific you are, the better I can help you. I am still learning, so sometimes I need a little help understanding what you're asking.
    - If you want to query your dataset, but your input is being interpreted as a documentation or general computer vision query, try using the `show` keyword. For example, `show me all images with a label of dog`.
    - If you want to query the FiftyOne documentation, try using either `docs` or `fiftyone` in your query. For example, `how do I load a dataset in fiftyone?`
    - If you want me to infer what you're asking based on our conversation history, try using the `now` keyword. For example, if you just asked "show me high confidence predictions of cats, dogs, and rabbits", you can ask "now the low confidence predictions".
    - To clear our chat history, you can use the `reset` keyword.
    - To exit our chat, you can use the `exit` keyword.

    **Learn more**
    - You can learn more about me and my capabilities by visiting my [GitHub page](https://github.com/voxel51/voxelgpt), and while you're at it, please give me a star! VoxelGPT is an open-source project, and it is constantly improving. If you'd like to contribute, check out the [Contributing](https://github.com/voxel51/voxelgpt#contributing) section of the README.
    - Learn more about [FiftyOne](https://github.com/voxel51/fiftyone), and give the project a ⭐! FiftyOne is open-source too!
    - Join the [FiftyOne Community Slack](https://slack.voxel51.com/), where thousands of computer vision enthusiasts and professionals are discussing the latest in computer vision and machine learning.
    
    I'm still learning, so I appreciate your patience!
    """

    str_message = md_message.replace("**", "")
    return {
        "string": str_message,
        "markdown": md_message,
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
