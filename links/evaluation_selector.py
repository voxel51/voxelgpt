import pandas as pd

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate

llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

#####################################################################

EVAL_EXAMPLE_TEMPLATE = """
Query: {query}
Available evaluations: {available_evals}
Selected evaluation: {selected_eval}\n
"""

EVAL_EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["query", "available_evals", "selected_eval"],
    template=EVAL_EXAMPLE_TEMPLATE,
)

EVAL_PROMPT_PREFIX = "Return the name of the evaluation run required to generate the DatasetView specified in the query, given available evaluation runs:\n" 
EVAL_PROMPT_SUFFIX = "Query: {query}\nAvailable evaluations: {available_evals}\nSelected evaluation:"
EVAL_PROMPT_INPUTS = ["query", "available_evals"]

TASK_RULES_FILE = {
    "default": "prompts/evaluation_task_rules.txt"
}

EXAMPLES_FILE = {
    "default": "examples/fiftyone_evaluation_selector_examples.csv",
}

class EvaluationSelector:
    """Class to select the correct evaluation run for a given query and dataset"""

    def __init__(self, dataset):
        self.dataset = dataset
        #self.set_run_type()
        self.run_type = 'default'
        self.task_rules_file = self.get_task_rules_file()
        self.examples_file = self.get_examples_file()
        
    def set_run_type(self):
        raise NotImplementedError("set_run_type method not implemented")
    
    def set_task_rules_file(self):
        raise NotImplementedError("set_task_rules_file method not implemented")
    
    def set_examples_file(self):
        raise NotImplementedError("set_examples_file method not implemented")
    
    def get_evaluation_info(self, brain_run):
        raise NotImplementedError("get_brain_run_info method not implemented")
    
    #OK
    def get_available_evaluations(self):
        evals = self.dataset.list_evaluations()
        eval_infos_reduced = [self.get_reduced_evaluation_info(x) for x in evals]
        return eval_infos_reduced
        #raise NotImplementedError("get_available_brain_runs method not implemented")

    #OK
    def get_reduced_evaluation_info(self, eval_key):
        info = self.dataset.get_evaluation_info(eval_key)
        cfg = info.config

        CONFIG_KEYS_KEEP = ['method', 'cls', 'pred_field', 'gt_field', 'iou']
        info_keep = {'key': eval_key}
        info_keep.update({f'config_{k}':getattr(cfg,k) for k in CONFIG_KEYS_KEEP if hasattr(cfg,k)})

        return info_keep
    
    def get_evaluation(self):
        raise NotImplementedError("get_brain_run method not implemented")
    
    def get_task_rules_file(self):
        return TASK_RULES_FILE[self.run_type]
    
    def get_examples_file(self):
        return EXAMPLES_FILE[self.run_type]
    
    def value_error(self):
        raise ValueError(f"No {self.run_type} runs found")
    
    def load_prompt_prefix(self):
        with open(self.task_rules_file, "r") as f:
            prompt_prefix = f.read() # + '\n'
        return prompt_prefix
    
    '''
    def load_prompt_suffix(self, query, brain_runs):
        return self.prompt_suffix.format(
            query=query,
            brain_runs=brain_runs
        )
    '''

    def generate_prompt(self, query, available_evals):
        prefix = self.load_prompt_prefix()
        prefix += ', '.join([x['key'] for x in available_evals])
        prefix += '\n'
        body = self.generate_examples_prompt(query, available_evals)
        return (prefix + body).replace('{', '(').replace('}', ')')
    
    def generate_examples_prompt(self, query, available_evals):
        examples = self.get_examples()

        return FewShotPromptTemplate(
            examples=examples,
            example_prompt=EVAL_EXAMPLE_PROMPT,
            prefix=EVAL_PROMPT_PREFIX,
            suffix=EVAL_PROMPT_SUFFIX,
            input_variables=EVAL_PROMPT_INPUTS,
            example_separator="\n",
        ).format(
            query=query, 
            available_evals=available_evals, #            run_type=self.run_type
            )
    
    def get_examples(self):
        with open(self.examples_file, "r") as f:
            df = pd.read_csv(f)
        examples = []

        for _, row in df.iterrows():
            example = {
                "query": row.query,
                "available_evals": row.available_evals,
                "selected_eval": row.selected_eval
            }
            examples.append(example)
        return examples
    
    def llm_select_evaluation(self, query, evals=None):
        if evals is None:
            evals = self.get_available_evaluations()

        prompt = self.generate_prompt(query, evals)
        response = llm.call_as_llm(prompt)
        return evals, prompt, response.strip()


    def select_evaluation_run(self, query):
        evals = self.get_available_evaluations()
        if len(evals) == 0:
            self.value_error()
        elif len(evals) == 1:
            return evals[0]['key']
        else:
            return self.llm_select_evaluation(query, evals=evals)

    
#OK
class DefaultEvaluationSelector(EvaluationSelector):
     def __init__(self, dataset):
        super().__init__(dataset)

#OK
def select_evaluation_run(dataset, query):
    selector = DefaultEvaluationSelector(dataset)
    return selector.select_evaluation_run(query)


'''
brain_run_selectors = {
    "uniqueness": UniquenessBrainRunSelector,
    "mistakenness": MistakennessBrainRunSelector,
    "image_similarity": ImageSimilarityBrainRunSelector,
    "text_similarity": TextSimilarityBrainRunSelector,
    # "hardness": HardnessBrainRunSelector
}
'''

'''
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
'''
























