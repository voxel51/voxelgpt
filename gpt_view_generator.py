"""
GPT view generator.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import fiftyone as fo

from links.query_validator import moderate_query
from links.query_intent_classifier import classify_query_intent
from links.docs_query_dispatcher import run_docs_query
from links.computer_vision_query_dispatcher import run_computer_vision_query
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
    if sample_collection.media_type not in ("image", "video"):
        raise Exception("Only image or video collections are supported")

    if chat_history is None:
        chat_history = []

    def _log(message):
        _log_chat_history(message, "GPT", chat_history)
        return message if raw else _emit_log(message)

    def _log_message_with_list(message, list):
        return _log(message + f"{', '.join(list)}")

    if not moderate_query(query):
        yield _log(
            "I'm sorry, this query does not abide by OpenAI's guidelines"
        )
        if chat_history:
            chat_history.pop()
        return

    # Continuing an existing conversation
    if len(chat_history) > 2:
        query = generate_effective_query(chat_history)

    # Intent classification
    intent = classify_query_intent(query)
    if intent == 'documentation':
        yield _log(run_docs_query(query))
        return
    elif intent == 'computer_vision':
        yield _log(run_computer_vision_query(query))
        return
    elif intent != 'display':
        yield _log("I'm sorry, I don't understand")
        return
    ### else intent == 'display' --> continue

    # Algorithms
    algorithms = select_algorithms(query)
    if algorithms:
        message = f"Identified potential algorithms: "
        yield _log_message_with_list(message, algorithms)

    # Runs
    runs = select_runs(sample_collection, query, algorithms)
    if runs:
        run_keys = {k: v["key"] for k, v in runs.items()}
        yield _log(f"Identified potential runs: {run_keys}")

    # Fields
    fields = select_fields(sample_collection, query)
    if fields:
        message = f"Identified potential fields: "
        yield _log_message_with_list(message, fields)

    # Label classes
    label_classes = select_label_classes(sample_collection, query, fields)
    if label_classes == "_CONFUSED_" and "text_similarity" not in runs:
        yield _log("I'm sorry, I don't understand")
        return
    elif label_classes == "_CONFUSED_":
        label_classes = {}

    if any(len(v) > 0 for v in label_classes.values()):
        _label_classes, _unmatched_classes = _format_label_classes(
            label_classes
        )
        yield _log_label_classes(_label_classes, chat_history, raw)

    examples = generate_view_stage_examples_prompt(
        sample_collection, query, runs, label_classes
    )
    view_stage_descriptions = generate_view_stage_descriptions_prompt(examples)

    # View stages
    view_stages = get_most_relevant_view_stages(examples)
    if view_stages:
        yield _log_view_stage_list(view_stages, chat_history, raw)

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
        yield _log_load_dataset_message(view_str, chat_history, raw)

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


def _get_stage_doc_link(stage_name):
    return f"https://docs.voxel51.com/api/fiftyone.core.collections.html#fiftyone.core.collections.SampleCollection.{stage_name}"


def _wrap_stage_name(stage_name):
    return f"[{stage_name}()]({_get_stage_doc_link(stage_name)})"


def _log_view_stage_list(view_stage_names, chat_history, raw):
    gpt_message = f"Identified potential view stages: "
    raw_message = gpt_message + f"{view_stage_names}"
    _log_chat_history(raw_message, "GPT", chat_history)

    if raw:
        return raw_message
    else:

        formatted_stage_names = [
            _wrap_stage_name(stage_name) for stage_name in view_stage_names
        ]

        formatted_message = gpt_message + f"{', '.join(formatted_stage_names)}"
        return _emit_log(formatted_message)


def _format_label_and_classes(label_field, class_names):
    prefix = f"  \n - **{label_field}**:"
    if class_names:
        return prefix + f" \t{', '.join(class_names)}"
    else:
        return prefix + f" \t*no classes found*"


def _log_label_classes(label_classes, chat_history, raw):
    gpt_message = f"Identified potential label classes: "
    raw_message = gpt_message + f"{label_classes}"
    _log_chat_history(raw_message, "GPT", chat_history)

    if raw:
        return raw_message
    else:
        formatted_label_classes = [
            _format_label_and_classes(k, v) for k, v in label_classes.items()
        ]

        formatted_message = (
            gpt_message + f"{' '.join(formatted_label_classes)}"
        )
        return _emit_log(formatted_message)


def _log_load_dataset_message(view_str, chat_history, raw):
    gpt_message = f"Okay, I'm going to load "
    raw_message = gpt_message + f"{view_str}"
    _log_chat_history(raw_message, "GPT", chat_history)

    if raw:
        return raw_message
    else:
        formatted_message = gpt_message + f"`{view_str}`"
        return _emit_log(formatted_message)


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


def _emit_log(message):
    return {"type": "log", "data": {"message": message}}


def _emit_view(view):
    return {"type": "view", "data": {"view": view}}
