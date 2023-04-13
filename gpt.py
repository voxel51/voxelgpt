from links.view_stage_example_selector import generate_view_stage_examples_prompt
from links.view_stage_description_selector import generate_view_stage_descriptions_prompt, get_most_relevant_view_stages
from links.brain_method_selector import select_brain_methods
from links.brain_run_selector import select_brain_runs
from links.field_selector import select_fields
from links.dataset_view_generator import get_gpt_view_stage_strings
from links.view_stage_validator import validate_view_stages

from fiftyone import ViewField as F

def verbose_print(verbose, string):
    if verbose:
        print(string)

def generate_view(query, dataset, verbose=False):
    verbose_print(verbose, f"Finding similar examples for query: {query}")
    examples = generate_view_stage_examples_prompt(
        dataset, query
        )
    view_stages = get_most_relevant_view_stages(examples)
    verbose_print(verbose, f"Identified likely view stages: {view_stages}")
    view_stage_descriptions = generate_view_stage_descriptions_prompt(examples)
    brain_methods = select_brain_methods(query)
    verbose_print(verbose, f"Identified brain methods: {brain_methods}")
    brain_runs = select_brain_runs(dataset, query, brain_methods)
    verbose_print(verbose, f"Identified brain runs: {brain_runs}")
    fields = select_fields(dataset, query)
    verbose_print(verbose, f"Identified fields: {fields}")

    verbose_print(verbose, "Converting query to list of view stages...")
    view_stages = get_gpt_view_stage_strings(
        dataset,
        brain_runs,
        fields,
        view_stage_descriptions,
        examples
    )

    verbose_print(verbose, f"Validating view stages...\n--->{view_stages}")
    validate_view_stages(view_stages)

    eval_string = f"dataset." + ".".join(view_stages) 
    view = dataset.view()
    view = eval(eval_string)
    return view
