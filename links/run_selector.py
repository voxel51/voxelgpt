import pandas as pd

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

#####################################################################

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

TASK_RULES_FILE = {
    "uniqueness": "prompts/uniqueness_task_rules.txt",
    "hardness": "prompts/hardness_task_rules.txt",
    "mistakenness": "prompts/mistakenness_task_rules.txt",
    "image_similarity": "prompts/image_similarity_task_rules.txt",
    "text_similarity": "prompts/text_similarity_task_rules.txt",
    "evaluation": "prompts/evaluation_task_rules.txt",
    "metadata": None,
}

EXAMPLES_FILE = {
    "uniqueness": "examples/fiftyone_uniqueness_run_examples.csv",
    "hardness": "examples/fiftyone_hardness_run_examples.csv",
    "mistakenness": "examples/fiftyone_mistakenness_run_examples.csv",
    "image_similarity": "examples/fiftyone_image_similarity_run_examples.csv",
    "text_similarity": "examples/fiftyone_text_similarity_run_examples.csv",
    "evaluation": "examples/fiftyone_evaluation_run_examples.csv",
    "metadata": None,
}


class RunSelector:
    """Class to select the correct run for a given query and dataset"""

    def __init__(self, dataset):
        self.dataset = dataset
        self.set_run_type()
        self.task_rules_file = self.get_task_rules_file()
        self.examples_file = self.get_examples_file()

    def set_run_type(self):
        raise NotImplementedError("set_run_type method not implemented")

    def set_task_rules_file(self):
        raise NotImplementedError("set_task_rules_file method not implemented")

    def set_examples_file(self):
        raise NotImplementedError("set_examples_file method not implemented")

    def get_run_info(self, run):
        raise NotImplementedError("get_run_info method not implemented")

    def get_available_runs(self):
        raise NotImplementedError("get_available_runs method not implemented")

    def print_compute_run_message(self):
        message = self.generate_compute_run_message()
        print(message)

    def get_run(self):
        raise NotImplementedError("get_run method not implemented")

    def get_task_rules_file(self):
        return TASK_RULES_FILE[self.run_type]

    def get_examples_file(self):
        return EXAMPLES_FILE[self.run_type]

    def value_error(self):
        raise ValueError(f"No {self.run_type} runs found")

    def load_prompt_prefix(self):
        with open(self.task_rules_file, "r") as f:
            prompt_prefix = f.read() + "\n"
        return prompt_prefix

    def load_prompt_suffix(self, query, runs):
        return self.prompt_suffix.format(query=query, runs=runs)

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
        with open(self.examples_file, "r") as f:
            df = pd.read_csv(f)
        examples = []

        for _, row in df.iterrows():
            example = {
                "query": row.query,
                "available_runs": row.available_runs,
                "selected_run": row.selected_run,
            }
            examples.append(example)
        return examples

    def select_run(self, query):
        available_runs = self.get_available_runs()
        if len(available_runs) == 0:
            self.print_compute_run_message()
            return None
        elif len(available_runs) == 1:
            return available_runs[0]
        else:
            prompt = self.generate_prompt(query, available_runs)
            response = llm.call_as_llm(prompt).strip()
            if response not in available_runs:
                response = available_runs[0]
            return response


class EvaluationRunSelector(RunSelector):
    """Class to select the correct evaluation run for a given query and dataset"""

    def __init__(self, dataset):
        super().__init__(dataset)

    def set_run_type(self):
        self.run_type = "evaluation"

    def generate_compute_run_message(self):
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
    """Class to select the correct uniqueness run for a given query and dataset"""

    def __init__(self, dataset):
        super().__init__(dataset)

    def set_run_type(self):
        self.run_type = "uniqueness"

    def generate_compute_run_message(self):
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
    """Class to select the correct mistakenness run for a given query and dataset"""

    def __init__(self, dataset):
        super().__init__(dataset)

    def set_run_type(self):
        self.run_type = "mistakenness"

    def generate_compute_run_message(self):
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
    """Class to select the correct image_similarity run for a given query and dataset"""

    def __init__(self, dataset):
        super().__init__(dataset)

    def set_run_type(self):
        self.run_type = "image_similarity"

    def generate_compute_run_message(self):
        message = "No similarity index found. To generate a similarity index for your samples, please run the following command:\n"
        command = """
        ```
        import fiftyone.brain as fob
        fob.compute_similarity(
            dataset, 
            model='mobilenet-v2-imagenet-torch',
            brain_run_key='img_sim',
            )
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
    """Class to select the correct text_similarity run for a given query and dataset"""

    def __init__(self, dataset):
        super().__init__(dataset)

    def set_run_type(self):
        self.run_type = "text_similarity"

    def generate_compute_run_message(self):
        message = "No similarity index found that supports text prompts. To generate a similarity index for your samples, please run the following command:\n"
        command = """
        ```
        import fiftyone.brain as fob
        fob.compute_similarity(
            dataset, 
            model='clip-vit-base32-torch',
            brain_run_key='text_sim',
            )
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
    """Class to select the correct hardness run for a given query and dataset"""

    def __init__(self, dataset):
        super().__init__(dataset)

    def set_run_type(self):
        self.run_type = "hardness"

    def generate_compute_run_message(self):
        message = "No hardness run found. To measure of the uncertainty of your model's predictions (in `<label_field>`) on the samples in your dataset, please run the following command:\n"
        command = """
        ```
        import fiftyone.brain as fob
        fob.compute_hardness(
            dataset, 
            <label_field>,
            )
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
    """Class for metadata computation validation"""

    def __init__(self, dataset):
        super().__init__(dataset)

    def set_run_type(self):
        self.run_type = "metadata"

    def generate_compute_run_message(self):
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
        nsamples = self.dataset.count()
        if self.dataset.exists("metadata").count() != nsamples:
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


class RunsSelector:
    """Class to select the correct runs for a given query and dataset"""

    def __init__(self, dataset):
        self.dataset = dataset

    def select_runs(self, query, run_types):
        selected_runs = {}

        for rt in run_types:
            run_selector = run_selectors[rt](self.dataset)
            run = run_selector.select_run(query)
            if run:
                selected_runs[rt] = run

        return selected_runs


def select_runs(dataset, query, run_types):
    runs_selector = RunsSelector(dataset)
    return runs_selector.select_runs(query, run_types)
