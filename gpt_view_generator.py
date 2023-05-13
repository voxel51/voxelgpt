"""
GPT view generator.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import traceback

import fiftyone as fo

from links.query_validator import validate_query
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


def ask_gpt_interactive(dataset, session=None):
    chat_history = []

    if session is None:
        session = fo.launch_app(dataset, auto=False)

    while True:
        prompt = "How can I help you?"
        if not chat_history:
            _log_chat_history(prompt, "GPT", chat_history)

        query = input(prompt + " ")

        if not query or query == "exit":
            break

        if query == "reset":
            chat_history.clear()
            continue

        _log_chat_history(query, "User", chat_history)

        view = ask_gpt(dataset, query, chat_history=chat_history)

        if view is not None:
            session.view = view


def ask_gpt(dataset, query, chat_history=None):
    for response in ask_gpt_generator(
        dataset, query, chat_history=chat_history
    ):
        type = response["type"]
        data = response["data"]

        if type == "view":
            return data["view"]
        elif type == "log":
            msg = data["message"]
            print(msg)
        elif type == "error":
            msg = data["message"]
            print("ERROR: %s" % msg)


def ask_gpt_generator(dataset, query, chat_history=None, raw=False):
    if chat_history is None:
        chat_history = []

    def _log(message):
        _log_chat_history(message, "GPT", chat_history)
        return message if raw else _emit_log(message)

    def _error(message, code=None):
        if raw:
            raise Exception(message)

        return _emit_error(
            message,
            trace=traceback.format_exc(),
            code=code,
        )

    if dataset.media_type not in ("image", "video"):
        yield _error("Only image and video datasets are supported", code=400)
        return

    # Continuing an existing conversation
    if len(chat_history) > 2:
        query = generate_effective_query(chat_history)

    if not validate_query(query):
        yield _log("I'm sorry, I don't understand")
        return

    examples = generate_view_stage_examples_prompt(dataset, query)
    view_stage_descriptions = generate_view_stage_descriptions_prompt(examples)

    # View stages
    view_stages = get_most_relevant_view_stages(examples)
    yield _log(f"Identified likely view stages: {view_stages}")

    # Algorithms
    algorithms = select_algorithms(query)
    if len(algorithms) > 0:
        yield _log(f"Identified algorithms: {algorithms}")

    # Runs
    runs = select_runs(dataset, query, algorithms)
    run_keys = {k: v["key"] for k, v in runs.items()}
    if len(runs) > 0:
        yield _log(f"Identified runs: {run_keys}")

    # Fields
    fields = select_fields(dataset, query)
    yield _log(f"Identified potentially relevant fields: {fields}")

    # Label classes
    label_classes = select_label_classes(dataset, query, fields)
    if label_classes == "_CONFUSED_":
        yield _log("I'm sorry, I don't understand")
        return

    if any(len(v) > 0 for v in label_classes.values()):
        _label_classes = _format_label_classes(label_classes)
        yield _log(f"Identified label classes: {_label_classes}")
    else:
        yield _log("Did not identify any relevant label classes")

    examples = _reformat_query(examples, label_classes)

    stages = get_gpt_view_stage_strings(
        dataset,
        runs,
        fields,
        _format_label_classes(label_classes),
        view_stage_descriptions,
        examples,
    )

    if raw:
        yield _log(stages)
        return

    if "metadata" in ".".join(stages) and "metadata" not in runs:
        stages = "_NEED_METADATA_"

    if stages == "_NEED_METADATA_":
        yield _log("Please compute metadata first")
        return

    if stages == "_MORE_":
        yield _log("Please be more specific")
        return

    if stages == "_CONFUSED_":
        yield _log("I'm sorry, I don't understand")
        return

    yield _log(_format_stages(stages))

    try:
        view = dataset.view()
        code = "view." + ".".join(stages)
        view = eval(code)
        yield _emit_view(view)
    except Exception as e:
        yield _error(
            "Failed to create view from stages: %s" % str(e), code=500
        )


def _log_chat_history(text, speaker, history):
    history.append(f"{speaker}: {text}")


def _format_stages(stages):
    stages_text = ""
    for i, stage in enumerate(stages):
        stages_text += f"Stage {i+1}: {stage}\n"

    return stages_text


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
        else:
            clarification = f" where by {k} I mean any of {v}"
            query += clarification

    example_lines[-2] = query
    return "\n".join(example_lines)


def _format_label_classes(label_classes):
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
                field_list += el_val

        label_class_dict[field] = field_list

    return label_class_dict


def _emit_error(message, code=None, trace=None):
    return {
        "type": "error",
        "data": {
            "code": code,
            "message": message,
            "trace": trace,
        },
    }


def _emit_log(message):
    return {"type": "log", "data": {"message": message}}


def _emit_view(view):
    return {"type": "view", "data": {"view": view}}
