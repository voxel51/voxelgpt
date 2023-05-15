"""
GPT view generator.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
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


def ask_gpt_interactive(sample_collection, session=None):
    chat_history = []

    if session is None:
        session = fo.launch_app(sample_collection, auto=False)

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

        response = ask_gpt(sample_collection, query, chat_history=chat_history)

        if isinstance(response, fo.DatasetView):
            session.view = response
        elif isinstance(response, fo.Dataset):
            session.dataset = response


def ask_gpt(sample_collection, query, chat_history=None):
    for response in ask_gpt_generator(
        sample_collection, query, chat_history=chat_history
    ):
        type = response["type"]
        data = response["data"]

        if type == "view":
            return data["view"]
        elif type == "log":
            print(data["message"])


def ask_gpt_generator(sample_collection, query, chat_history=None, raw=False):
    if chat_history is None:
        chat_history = []

    if sample_collection.media_type not in ("image", "video"):
        raise Exception("Only image or video collections are supported")

    def _log(message):
        _log_chat_history(message, "GPT", chat_history)
        return message if raw else _emit_log(message)

    # Continuing an existing conversation
    if len(chat_history) > 2:
        query = generate_effective_query(chat_history)

    if not validate_query(query):
        yield _log("I'm sorry, I don't understand")
        return

    # Algorithms
    algorithms = select_algorithms(query)
    if algorithms:
        yield _log(f"Identified potential algorithms: {algorithms}")

    # Runs
    runs = select_runs(sample_collection, query, algorithms)
    if runs:
        run_keys = {k: v["key"] for k, v in runs.items()}
        yield _log(f"Identified potential runs: {run_keys}")

    # Fields
    fields = select_fields(sample_collection, query)
    if fields:
        yield _log(f"Identified potential fields: {fields}")

    # Label classes
    label_classes = select_label_classes(sample_collection, query, fields)
    if label_classes == "_CONFUSED_" and "text_similarity" not in runs:
        yield _log("I'm sorry, I don't understand")
        return
    elif label_classes == "_CONFUSED_":
        label_classes = {}

    if any(len(v) > 0 for v in label_classes.values()):
        _label_classes = _format_label_classes(label_classes)
        yield _log(f"Identified potential label classes: {_label_classes}")
    
    # _label_fields = list(label_classes.keys())

    examples = generate_view_stage_examples_prompt(
        sample_collection, 
        query,
        runs,
        label_classes
        )
    view_stage_descriptions = generate_view_stage_descriptions_prompt(examples)

    # View stages
    view_stages = get_most_relevant_view_stages(examples)
    if view_stages:
        yield _log(f"Identified potential view stages: {view_stages}")

    examples = _reformat_query(examples, label_classes)

    stages = get_gpt_view_stage_strings(
        sample_collection,
        runs,
        fields,
        _format_label_classes(label_classes),
        view_stage_descriptions,
        examples,
    )

    if "metadata" in ".".join(stages) and "metadata" not in runs:
        stages = "_NEED_METADATA_"

    if raw:
        yield stages
        return

    if stages == "_NEED_METADATA_":
        yield _log("Please compute metadata first")
        return

    if stages == "_MORE_":
        yield _log("Please be more specific")
        return

    if stages == "_CONFUSED_":
        yield _log("I'm sorry, I don't understand")
        return

    view_str = _build_view_str(sample_collection, stages)

    if view_str:
        yield _log("Okay, I'm going to load %s" % view_str)

        try:
            view = _build_view(sample_collection, view_str)
            yield _emit_view(view)
        except Exception as e:
            yield _log("Looks like the view was invalid. Please try again")
    else:
        if isinstance(sample_collection, fo.DatasetView):
            yield _log("I'm returning your entire view")
        else:
            yield _log("I'm returning your entire dataset")

        yield _emit_view(sample_collection)


def _build_view_str(sample_collection, stages):
    stages_str = ".".join(stages)
    if not stages_str:
        return None

    if isinstance(sample_collection, fo.DatasetView):
        return "view." + stages_str

    return "dataset." + stages_str


def _build_view(sample_collection, view_str):
    import fiftyone as fo
    from fiftyone import ViewField as F

    if isinstance(sample_collection, fo.DatasetView):
        view = sample_collection
        dataset = view._root_dataset
    else:
        dataset = sample_collection
        view = dataset.view()

    return eval(view_str)


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
        elif v:
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


def _emit_log(message):
    return {"type": "log", "data": {"message": message}}


def _emit_view(view):
    return {"type": "view", "data": {"view": view}}
