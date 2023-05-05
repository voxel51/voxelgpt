from links.query_validator import validate_query
from links.view_stage_example_selector import generate_view_stage_examples_prompt
from links.view_stage_description_selector import generate_view_stage_descriptions_prompt, get_most_relevant_view_stages
from links.algorithm_selector import select_algorithms
from links.run_selector import select_runs
from links.field_selector import select_fields
from links.label_class_selector import select_label_classes
from links.dataset_view_generator import get_gpt_view_stage_strings

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
    lcls = {}
    for field in label_fields:
        field_list = []
        lcl = label_classes[field]
        for el in lcl:
            el_val = list(el.values())[0]
            if type(el_val) == str:
                field_list.append(el_val)
            else:
                field_list += el_val
        lcls[field] = field_list
    return lcls


def get_gpt_view_text(dataset, query):
    #### Validate media type
    if dataset.media_type not in ["image", "video"]:
        print(f"At present, the FiftyOne GPT integration only supports image and video datasets. The dataset {dataset.name} has media type {dataset.media_type}. If you would like to use this feature, please try a different dataset.")
        return
    
    valid = validate_query(query)
    if not valid:
        return '_CONFUSED_'
    
    print(f"Finding similar examples for query: {query}")
    examples = generate_view_stage_examples_prompt(
        dataset, query
        )
    view_stages = get_most_relevant_view_stages(examples)
    print(f"Identified likely view stages: {view_stages}")
    view_stage_descriptions = generate_view_stage_descriptions_prompt(examples)
    algorithms = select_algorithms(query)
    if len(algorithms) > 0:
        print(f"Identified algorithms: {algorithms}")
    runs = select_runs(dataset, query, algorithms)
    run_keys = {k: v["key"] for k, v in runs.items()}
    if len(runs) > 0:
        print(f"Identified runs: {run_keys}")
    fields = select_fields(dataset, query)
    print(f"Identified potentially relevant fields: {fields}")
    label_classes = select_label_classes(dataset, query, fields)
    if label_classes == "_CONFUSED_":
        return "_CONFUSED_"
    lens = [len(v) for v in label_classes.values()]
    if any([l > 0 for l in lens]):
        print(
            f"Identified label classes: {format_label_classes(label_classes)}"
            )
    else:
        print(f"Did not identify any relevant label classes")
    
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