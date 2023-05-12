from links.query_validator import validate_query
from links.view_stage_example_selector import generate_view_stage_examples_prompt
from links.view_stage_description_selector import generate_view_stage_descriptions_prompt, get_most_relevant_view_stages
from links.algorithm_selector import select_algorithms
from links.run_selector import select_runs
from links.field_selector import select_fields
from links.label_class_selector import select_label_classes
from links.dataset_view_generator import get_gpt_view_stage_strings
from links.effective_query_generator import generate_effective_query

import fiftyone as fo
from fiftyone import ViewField as F

def log_chat_history(text, speaker, history):
    history.append(f"{speaker}: {text}")

def log_and_print_chat_history(text, speaker, history):
    log_chat_history(text, speaker, history)
    print(text)

def format_stages(stages):
    stages_text = ""
    for i, stage in enumerate(stages):
        stages_text += f"Stage {i+1}: {stage}\n"
    return stages_text

def reformat_query(examples, label_classes):
    example_lines = examples.split('\n')
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
    return '\n'.join(example_lines)

def format_label_classes(label_classes):
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


def get_gpt_view_text(dataset, query, chat_history):
    #### Validate media type
    if dataset.media_type not in ["image", "video"]:
        print(f"At present, the FiftyOne GPT integration only supports image and video datasets. The dataset {dataset.name} has media type {dataset.media_type}. If you would like to use this feature, please try a different dataset.")
        return

    valid = validate_query(query)
    if not valid:
        return '_CONFUSED_'

    examples = generate_view_stage_examples_prompt(
        dataset, query
        )

    view_stages = get_most_relevant_view_stages(examples)
    likely_view_stages_text = f"Identified likely view stages: {view_stages}"
    log_and_print_chat_history(likely_view_stages_text, "GPT", chat_history)
    view_stage_descriptions = generate_view_stage_descriptions_prompt(examples)
    algorithms = select_algorithms(query)
    if len(algorithms) > 0:
        algs_text = f"Identified algorithms: {algorithms}"
        log_and_print_chat_history(algs_text, "GPT", chat_history)
    runs = select_runs(dataset, query, algorithms)
    run_keys = {k: v["key"] for k, v in runs.items()}
    if len(runs) > 0:
        runs_text = f"Identified runs: {run_keys}"
        log_and_print_chat_history(runs_text, "GPT", chat_history)
    fields = select_fields(dataset, query)
    print(f"Identified potentially relevant fields: {fields}")
    label_classes = select_label_classes(dataset, query, fields)
    lens = [len(v) for v in label_classes.values()]
    if any([l > 0 for l in lens]):
        label_classes_text = f"Identified label classes: {format_label_classes(label_classes)}"
        log_and_print_chat_history(label_classes_text, "GPT", chat_history)

    else:
        label_classes_text = f"Did not identify any relevant label classes"
        log_and_print_chat_history(label_classes_text, "GPT", chat_history)

    examples = reformat_query(examples, label_classes)

    response = get_gpt_view_stage_strings(
        dataset,
        runs,
        fields,
        label_classes,
        view_stage_descriptions,
        examples
    )

    if "metadata" in ''.join(response) and "metadata" not in list(run_keys.keys()):
        return "_NEED_METADATA_"

    return response

def create_view_from_stages(stages, dataset, session, chat_history):
    log_and_print_chat_history(format_stages(stages), "GPT", chat_history)
    view = dataset.view()
    code = 'view.' + '.'.join(stages)
    try:
        view = eval(code)
        session.view = view
    except:
        view = None
        invalid_view_text = f"Attempted to create view from stages, but resulted in invalid view. Please try again."
        log_and_print_chat_history(invalid_view_text, "GPT", chat_history)

from IPython.display import clear_output

def gpt(dataset):
    chat_history = []
    session = fo.launch_app(dataset, auto = False)
    while True:
        clear_output(True)
        input_text = "Hello! I'm here to help you explore your datasets.\nMy reponses are based off of our chat history. To clear my history and restart, enter 'reset'.\nHow can I help you? "
        if len(chat_history) == 0:
            log_chat_history(input_text, "GPT", chat_history)
        query = input(input_text)
        log_chat_history(query, "User", chat_history)
        if query == "exit" or query == '':
            break
        if query == "reset":
            chat_history = []
            continue

        if len(chat_history) != 2:
            new_query = generate_effective_query(chat_history)
            if validate_query(new_query):
                query = new_query
                print(f"Effective query: {query}")

        stages = get_gpt_view_text(dataset, query, chat_history)

        if stages == "_MORE_":
            print("Please be more specific")
            continue
        if stages == "_CONFUSED_":
            print("I'm sorry, I don't understand")
            continue
        if stages == "_NEED_METADATA_":
            print("Please compute metadata first")
            continue

        create_view_from_stages(stages, dataset, session, chat_history)

    return
