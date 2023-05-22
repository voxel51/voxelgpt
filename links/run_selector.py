"""
Run selector.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

from langchain.prompts import PromptTemplate, FewShotPromptTemplate
import pandas as pd

# pylint: disable=relative-beyond-top-level
from .utils import get_llm, get_cache


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")
PROMPTS_DIR = os.path.join(ROOT_DIR, "prompts")

RUN_EXAMPLE_TEMPLATE = """
Query: {query}
Available runs: {available_runs}
Selected run: {selected_run}\n
"""

RUN_EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["query", "available_runs", "selected_run"],
    template=RUN_EXAMPLE_TEMPLATE,
)

RUN_PROMPT_PREFIX = "Return the name of the {run_type} run required to generate the DatasetView specified in the query, given available {run_type} runs:\n"
RUN_PROMPT_SUFFIX = (
    "Query: {query}\nAvailable runs: {available_runs}\nSelected run:"
)
RUN_PROMPT_INPUTS = ["run_type", "query", "available_runs"]

TASK_RULES_PATHS = {
    "uniqueness": os.path.join(PROMPTS_DIR, "uniqueness_task_rules.txt"),
    "hardness": os.path.join(PROMPTS_DIR, "hardness_task_rules.txt"),
    "mistakenness": os.path.join(PROMPTS_DIR, "mistakenness_task_rules.txt"),
    "image_similarity": os.path.join(
        PROMPTS_DIR, "image_similarity_task_rules.txt"
    ),
    "text_similarity": os.path.join(
        PROMPTS_DIR, "text_similarity_task_rules.txt"
    ),
    "evaluation": os.path.join(PROMPTS_DIR, "evaluation_task_rules.txt"),
    "metadata": None,
}

EXAMPLES_PATHS = {
    "uniqueness": os.path.join(
        EXAMPLES_DIR, "fiftyone_uniqueness_run_examples.csv"
    ),
    "hardness": os.path.join(
        EXAMPLES_DIR, "fiftyone_hardness_run_examples.csv"
    ),
    "mistakenness": os.path.join(
        EXAMPLES_DIR, "fiftyone_mistakenness_run_examples.csv"
    ),
    "image_similarity": os.path.join(
        EXAMPLES_DIR, "fiftyone_image_similarity_run_examples.csv"
    ),
    "text_similarity": os.path.join(
        EXAMPLES_DIR, "fiftyone_text_similarity_run_examples.csv"
    ),
    "evaluation": os.path.join(
        EXAMPLES_DIR, "fiftyone_evaluation_run_examples.csv"
    ),
    "metadata": None,
}


class RunSelector(object):
    """Interface for selecting the correct run for a given query and dataset."""

    def __init__(self, sample_collection):
        self.sample_collection = sample_collection
        self.task_rules_path = TASK_RULES_PATHS[self.run_type]
        self.examples_path = EXAMPLES_PATHS[self.run_type]

    @property
    def dataset(self):
        return self.sample_collection._root_dataset

    @property
    def run_type(self):
        raise NotImplementedError("run_type not implemented")

    def get_available_runs(self):
        raise NotImplementedError("get_available_runs() not implemented")

    def get_run_info(self, run):
        raise NotImplementedError("get_run_info() not implemented")

    def get_run(self):
        raise NotImplementedError("get_run() not implemented")

    def compute_run_message(self):
        raise NotImplementedError("compute_run_message() not implemented")

    def load_prompt_prefix(self):
        with open(self.task_rules_path, "r") as f:
            return f.read() + "\n"

    def generate_prompt(self, query, runs):
        prefix = self.load_prompt_prefix()
        body = self.generate_examples_prompt(query, runs)
        return (prefix + body).replace("{", "(").replace("}", ")")

    def generate_examples_prompt(self, query, available_runs):
        examples = self.get_examples()

        return FewShotPromptTemplate(
            examples=examples,
            example_prompt=RUN_EXAMPLE_PROMPT,
            prefix=RUN_PROMPT_PREFIX,
            suffix=RUN_PROMPT_SUFFIX,
            input_variables=RUN_PROMPT_INPUTS,
            example_separator="\n",
        ).format(
            query=query, available_runs=available_runs, run_type=self.run_type
        )

    def get_examples(self):
        cache = get_cache()
        key = self.run_type + "_examples"
        if key not in cache:
            with open(self.examples_path, "r") as f:
                df = pd.read_csv(f)

            examples = []

            for _, row in df.iterrows():
                example = {
                    "query": row.query,
                    "available_runs": row.available_runs,
                    "selected_run": row.selected_run,
                }
                examples.append(example)
            
            cache[key] = examples
        
        return cache[key]

    def select_run(self, query):
        available_runs = self.get_available_runs()
        if not available_runs:
            print(self.compute_run_message())
            return None

        if len(available_runs) == 1:
            return available_runs[0]

        prompt = self.generate_prompt(query, available_runs)
        response = get_llm().call_as_llm(prompt).strip()
        if response not in available_runs:
            response = available_runs[0]

        return response


class EvaluationRunSelector(RunSelector):
    """Selects the correct evaluation run for a given query and dataset."""

    @property
    def run_type(self):
        return "evaluation"

    def compute_run_message(self):
        return "No evaluation runs found."
        # base_message =  "No evaluation runs found.\n\n"
        # detection_message = "If you want to compute detection evaluation, please run the following command:\n"
        # detection_command = """
        # ```
        # dataset.evaluate_detections(
        #     "<det_predictions>",
        #     eval_key="eval_det",
        # )
        # ```
        # """

        # classification_message = "If you want to compute classification evaluation, please run the following command:\n"
        # classification_command = """
        # ```
        # dataset.evaluate_classifications(
        #     "<classif_predictions>",
        #     eval_key="eval_classif",
        # )
        # ```
        # """
        # message = base_message + detection_message + detection_command + classification_message + classification_command
        # return message

    def get_run_info(self, run):
        key = run.key
        config = run.config

        dict = {"key": key}

        try:
            dict["method"] = config.method
        except:
            pass

        try:
            dict["pred_field"] = config.pred_field
        except:
            pass

        try:
            dict["gt_field"] = config.gt_field
        except:
            pass

        try:
            dict["iou"] = config.iou
        except:
            pass

        return dict

    def get_available_runs(self):
        runs = self.dataset.list_evaluations()
        runs = [self.dataset.get_evaluation_info(run) for run in runs]
        runs = [self.get_run_info(run) for run in runs]
        return runs


class UniquenessRunSelector(RunSelector):
    """Selects the correct uniqueness run for a given query and dataset."""

    @property
    def run_type(self):
        return "uniqueness"

    def compute_run_message(self):
        message = "No uniqueness runs found. If you want to compute uniqueness, please run the following command:\n"
        command = """
        ```
        import fiftyone.brain as fob
        fob.compute_uniqueness(dataset)
        ```
        """
        return message + command

    def get_run_info(self, run):
        key = run.key
        model = run.config.model.split(".")[-1]
        uniqueness_field = run.config.uniqueness_field
        return {
            "key": key,
            "model": model,
            "uniqueness_field": uniqueness_field,
        }

    def get_available_runs(self):
        runs = self.dataset.list_brain_runs(method="uniqueness")
        runs = [self.dataset.get_brain_info(r) for r in runs]
        runs = [self.get_run_info(r) for r in runs]
        return runs


class MistakennessRunSelector(RunSelector):
    """Selects the correct mistakenness run for a given query and dataset."""

    @property
    def run_type(self):
        return "mistakenness"

    def compute_run_message(self):
        message = "No mistakenness runs found. To compute the difficulty of classifying samples (`<pred_field>`) with respect to ground truth label `<gt_field>`, please run the following command:\n"
        command = """
        ```
        import fiftyone.brain as fob
        fob.compute_mistakenness(
            dataset,
            <pred_field>,
            label_field=<gt_field>
            )
        ```
        """
        return message + command

    def get_run_info(self, run):
        key = run.key
        prediction_field = run.config.pred_field
        label_field = run.config.label_field
        mistakenness_field = run.config.mistakenness_field
        return {
            "key": key,
            "mistakenness_field": mistakenness_field,
            "prediction_field": prediction_field,
            "label_field": label_field,
        }

    def get_available_runs(self):
        runs = self.dataset.list_brain_runs(method="mistakenness")
        runs = [self.dataset.get_brain_info(r) for r in runs]
        runs = [self.get_run_info(r) for r in runs]
        return runs


class ImageSimilarityRunSelector(RunSelector):
    """Selects the correct image similarity run for a given query and dataset."""

    @property
    def run_type(self):
        return "image_similarity"

    def compute_run_message(self):
        message = "No similarity index found. To generate a similarity index for your samples, please run the following command:\n"
        command = """
        ```
        import fiftyone.brain as fob
        fob.compute_similarity(dataset, brain_key="img_sim")
        ```
        """
        return message + command

    def get_run_info(self, run):
        key = run.key
        method = run.config.method
        embeddings_field = run.config.embeddings_field
        model = run.config.model
        patches_field = run.config.patches_field
        return {
            "key": key,
            "method": method,
            "embeddings_field": embeddings_field,
            "model": model,
            "patches_field": patches_field,
        }

    def get_available_runs(self):
        runs = []

        for run in self.dataset.list_brain_runs():
            info = self.dataset.get_brain_info(run)
            if "Similarity" in info.config.cls:
                runs.append(info)

        runs = [self.get_run_info(r) for r in runs]
        return runs


class TextSimilarityRunSelector(RunSelector):
    """Selects the correct text similarity run for a given query and dataset."""

    @property
    def run_type(self):
        return "text_similarity"

    def compute_run_message(self):
        message = "No similarity index found that supports text prompts. To generate a similarity index for your samples, please run the following command:\n"
        command = """
        ```
        import fiftyone.brain as fob
        fob.compute_similarity(dataset, model="clip-vit-base32-torch", brain_key="text_sim")
        ```
        """
        return message + command

    def get_run_info(self, run):
        key = run.key
        method = run.config.method
        model = run.config.model
        patches_field = run.config.patches_field
        return {
            "key": key,
            "backend": method,
            "model": model,
            "patches_field": patches_field,
        }

    def get_available_runs(self):
        runs = []

        for run in self.dataset.list_brain_runs():
            info = self.dataset.get_brain_info(run)
            if (
                "Similarity" in info.config.cls
                and info.config.supports_prompts
            ):
                runs.append(info)

        runs = [self.get_run_info(r) for r in runs]
        return runs


class HardnessRunSelector(RunSelector):
    """Selects the correct hardness run for a given query and dataset."""

    @property
    def run_type(self):
        return "hardness"

    def compute_run_message(self):
        message = "No hardness run found. To measure of the uncertainty of your model's predictions (in `label_field`) on the samples in your dataset, please run the following command:\n"
        command = """
        ```
        import fiftyone.brain as fob
        fob.compute_hardness(dataset, label_field)
        ```
        """
        return message + command

    def get_run_info(self, run):
        key = run.key
        label_field = run.config.label_field
        hardness_field = run.config.hardness_field
        return {
            "key": key,
            "label_field": label_field,
            "hardness_field": hardness_field,
        }

    def get_available_runs(self):
        runs = self.dataset.list_brain_runs(method="hardness")
        runs = [self.dataset.get_brain_info(r) for r in runs]
        runs = [self.get_run_info(r) for r in runs]
        return runs


class MetadataRunSelector(RunSelector):
    """Class for metadata computation validation."""

    @property
    def run_type(self):
        return "metadata"

    def compute_run_message(self):
        message = "No metadata found. To compute metadata for your samples, please run the following command:\n"
        command = """
        ```
        dataset.compute_metadata()
        ```
        """
        return message + command

    def get_run_info(self, run):
        return {"key": "metadata"}

    def get_available_runs(self):
        nsamples = self.sample_collection.count()
        if self.sample_collection.exists("metadata").count() != nsamples:
            return []
        else:
            return [self.get_run_info("metadata")]


run_selectors = {
    "uniqueness": UniquenessRunSelector,
    "mistakenness": MistakennessRunSelector,
    "image_similarity": ImageSimilarityRunSelector,
    "text_similarity": TextSimilarityRunSelector,
    "hardness": HardnessRunSelector,
    "evaluation": EvaluationRunSelector,
    "metadata": MetadataRunSelector,
}


class RunsSelector(object):
    """Selects the correct runs for a given query and dataset."""

    def __init__(self, sample_collection):
        self.sample_collection = sample_collection

    def select_runs(self, query, run_types):
        selected_runs = {}
        for rt in run_types:
            run_selector = run_selectors[rt](self.sample_collection)
            run = run_selector.select_run(query)
            if run:
                selected_runs[rt] = run

        return selected_runs


def select_runs(sample_collection, query, run_types):
    runs_selector = RunsSelector(sample_collection)
    return runs_selector.select_runs(query, run_types)
