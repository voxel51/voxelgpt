import pandas as pd

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate

llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

#####################################################################

BRAIN_RUN_EXAMPLE_TEMPLATE = """
Query: {query}
Available runs: {available_runs}
Selected run: {selected_run}\n
"""

BRAIN_RUN_EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["query", "available_runs", "selected_run"],
    template=BRAIN_RUN_EXAMPLE_TEMPLATE,
)

BRAIN_RUN_PROMPT_PREFIX = "Return the name of the {run_type} run required to generate the DatasetView specified in the query, given available {run_type} runs:\n" 
BRAIN_RUN_PROMPT_SUFFIX = "Query: {query}\nAvailable runs: {available_runs}\nSelected run:"
BRAIN_RUN_PROMPT_INPUTS = ["run_type", "query", "available_runs"]

TASK_RULES_FILE = {
    "uniqueness": "prompts/uniqueness_task_rules.txt",
    "hardness": "prompts/hardness_task_rules.txt",
    "mistakenness": "prompts/mistakenness_task_rules.txt",
    "image_similarity": "prompts/image_similarity_task_rules.txt",
    "text_similarity": "prompts/text_similarity_task_rules.txt",
}

EXAMPLES_FILE = {
    "uniqueness": "examples/fiftyone_uniqueness_run_examples.csv",
    "hardness": "examples/fiftyone_hardness_run_examples.csv",
    "mistakenness": "examples/fiftyone_mistakenness_run_examples.csv",
    "image_similarity": "examples/fiftyone_image_similarity_run_examples.csv",
    "text_similarity": "examples/fiftyone_text_similarity_run_examples.csv",
}

class BrainRunSelector:
    """Class to select the correct brain run for a given query and dataset"""

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
    
    def get_brain_run_info(self, brain_run):
        raise NotImplementedError("get_brain_run_info method not implemented")
    
    def get_available_brain_runs(self):
        raise NotImplementedError("get_available_brain_runs method not implemented")

    def get_brain_run(self):
        raise NotImplementedError("get_brain_run method not implemented")
    
    def get_task_rules_file(self):
        return TASK_RULES_FILE[self.run_type]
    
    def get_examples_file(self):
        return EXAMPLES_FILE[self.run_type]
    
    def value_error(self):
        raise ValueError(f"No {self.run_type} runs found")
    
    def load_prompt_prefix(self):
        with open(self.task_rules_file, "r") as f:
            prompt_prefix = f.read() + '\n'
        return prompt_prefix
    
    def load_prompt_suffix(self, query, brain_runs):
        return self.prompt_suffix.format(
            query=query,
            brain_runs=brain_runs
        )
    
    def generate_prompt(self, query, brain_runs):
        prefix = self.load_prompt_prefix()
        body = self.generate_examples_prompt(query, brain_runs)
        return (prefix + body).replace('{', '(').replace('}', ')')
    
    def generate_examples_prompt(self, query, available_runs):
        examples = self.get_examples()

        return FewShotPromptTemplate(
            examples=examples,
            example_prompt=BRAIN_RUN_EXAMPLE_PROMPT,
            prefix=BRAIN_RUN_PROMPT_PREFIX,
            suffix=BRAIN_RUN_PROMPT_SUFFIX,
            input_variables=BRAIN_RUN_PROMPT_INPUTS,
            example_separator="\n",
        ).format(
            query=query, 
            available_runs=available_runs, 
            run_type=self.run_type
            )
    
    def get_examples(self):
        with open(self.examples_file, "r") as f:
            df = pd.read_csv(f)
        examples = []

        for _, row in df.iterrows():
            example = {
                "query": row.query,
                "available_runs": row.available_runs,
                "selected_run": row.selected_run
            }
            examples.append(example)
        return examples
    
    def select_brain_run(self, query):
        available_brain_runs = self.get_available_brain_runs()
        if len(available_brain_runs) == 0:
            self.value_error()
        elif len(available_brain_runs) == 1:
            return available_brain_runs[0]['key']
        else:
            prompt = self.generate_prompt(query, available_brain_runs)
            response = llm.call_as_llm(prompt)
            return response.strip()


    
class UniquenessBrainRunSelector(BrainRunSelector):
    """Class to select the correct uniqueness brain run for a given query and dataset"""

    def __init__(self, dataset):
        super().__init__(dataset)

    def set_run_type(self):
        self.run_type = "uniqueness"
        
    def get_brain_run_info(self, brain_run):
        key = brain_run.key
        model = brain_run.config.model.split('.')[-1]
        uniqueness_field = brain_run.config.uniqueness_field
        return {"key": key, "model": model, "uniqueness_field": uniqueness_field}
    
    def get_available_brain_runs(self):
        brain_runs = self.dataset.list_brain_runs(method = "uniqueness")
        brain_runs = [self.dataset.get_brain_info(br) for br in brain_runs]
        brain_runs = [self.get_brain_run_info(br) for br in brain_runs]
        return brain_runs
    
class MistakennessBrainRunSelector(BrainRunSelector):
    """Class to select the correct mistakenness brain run for a given query and dataset"""

    def __init__(self, dataset):
        super().__init__(dataset)

    def set_run_type(self):
        self.run_type = "mistakenness"
        
    def get_brain_run_info(self, brain_run):
        key = brain_run.key
        prediction_field = brain_run.config.pred_field
        label_field = brain_run.config.label_field
        mistakenness_field = brain_run.config.mistakenness_field
        return {
            "key": key,
            "mistakenness_field": mistakenness_field,
            "prediction_field": prediction_field, 
            "label_field": label_field
            }
    
    def get_available_brain_runs(self):
        brain_runs = self.dataset.list_brain_runs(method = "mistakenness")
        brain_runs = [self.dataset.get_brain_info(br) for br in brain_runs]
        brain_runs = [self.get_brain_run_info(br) for br in brain_runs]
        return brain_runs
    
class ImageSimilarityBrainRunSelector(BrainRunSelector):
    """Class to select the correct image_similarity brain run for a given query and dataset"""

    def __init__(self, dataset):
        super().__init__(dataset)

    def set_run_type(self):
        self.run_type = "image_similarity"
        
    def get_brain_run_info(self, brain_run):
        key = brain_run.key
        method = brain_run.config.method
        embeddings_field = brain_run.config.embeddings_field
        model = brain_run.config.model
        patches_field = brain_run.config.patches_field
        return {
            "key": key, 
            "method": method, 
            "embeddings_field": embeddings_field,
            "model": model,
            "patches_field": patches_field
            }
    
    def get_available_brain_runs(self):
        brain_runs = []

        for run in self.dataset.list_brain_runs():
            info = self.dataset.get_brain_info(run)
            if "Similarity" in info.config.cls:
                brain_runs.append(info)

        brain_runs = [self.get_brain_run_info(br) for br in brain_runs]
        return brain_runs
    
class TextSimilarityBrainRunSelector(BrainRunSelector):
    """Class to select the correct text_similarity brain run for a given query and dataset"""

    def __init__(self, dataset):
        super().__init__(dataset)

    def set_run_type(self):
        self.run_type = "text_similarity"
        
    def get_brain_run_info(self, brain_run):
        key = brain_run.key
        method = brain_run.config.method
        model = brain_run.config.model
        patches_field = brain_run.config.patches_field
        return {
            "key": key, 
            "backend": method, 
            "model": model,
            "patches_field": patches_field
            }
    
    def get_available_brain_runs(self):
        brain_runs = []

        for run in self.dataset.list_brain_runs():
            info = self.dataset.get_brain_info(run)
            if "Similarity" in info.config.cls and info.config.supports_prompts:
                brain_runs.append(info)

        brain_runs = [self.get_brain_run_info(br) for br in brain_runs]
        return brain_runs
    
    
# class HardnessBrainRunSelector(BrainRunSelector):
#     """Class to select the correct hardness brain run for a given query and dataset"""

#     def __init__(self, dataset):
#         super().__init__(dataset)

#     def set_run_type(self):
#         self.run_type = "hardness"
        
#     ## TODO: add a method to get the available brain runs for a given dataset
#     def get_brain_run_info(self, brain_run):
#         field = brain_run.config.uniqueness_field
#         model = brain_run.config.model.split('.')[-1]
#         return {"key": field, "model": model}
    
#     def get_available_brain_runs(self):
#         brain_runs = self.dataset.list_brain_runs(method = "hardness")
#         brain_runs = [self.dataset.get_brain_info(br) for br in brain_runs]
#         brain_runs = [self.get_brain_run_info(br) for br in brain_runs]
#         return brain_runs
    

brain_run_selectors = {
    "uniqueness": UniquenessBrainRunSelector,
    "mistakenness": MistakennessBrainRunSelector,
    "image_similarity": ImageSimilarityBrainRunSelector,
    "text_similarity": TextSimilarityBrainRunSelector,
    # "hardness": HardnessBrainRunSelector
}


class BrainRunsSelector:
    """Class to select the correct brain runs for a given query and dataset"""

    def __init__(self, dataset):
        self.dataset = dataset

    def select_brain_runs(self, query, run_types):
        selected_runs = {}

        for rt in run_types:
            brain_run_selector = brain_run_selectors[rt](self.dataset)
            brain_run = brain_run_selector.select_brain_run(query)
            selected_runs[rt] = brain_run

        return selected_runs

def select_brain_runs(dataset, query, run_types):
    brain_runs_selector = BrainRunsSelector(dataset)
    return brain_runs_selector.select_brain_runs(query, run_types)

























