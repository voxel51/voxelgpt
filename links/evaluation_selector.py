import pandas as pd

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate

llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

#####################################################################

EVAL_EXAMPLE_TEMPLATE = """
Query: {query}
Available evaluations: {available_evaluations}
Selected evaluation: {selected_evaluation}\n
"""

EVAL_EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["query", "available_evaluations", "selected_evaluation"],
    template=EVAL_EXAMPLE_TEMPLATE,
)

EVAL_PROMPT_PREFIX = "Return the name of the evaluation run required to generate the DatasetView specified in the query, given available evaluation runs:\n" 
EVAL_PROMPT_SUFFIX = "Query: {query}\nAvailable evaluations: {available_evals}\nSelected evaluation:"
EVAL_PROMPT_INPUTS = ["query", "available_evals"]

TASK_RULES_FILE = {
    "default": "prompts/evaluation_task_rules.txt"
}

EXAMPLES_FILE = {
    "default": "examples/fiftyone_evaluation_selector_examples_3.csv",
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
    
    def get_available_evaluations(self):
        evals = self.dataset.list_evaluations()
        eval_infos_reduced = [self.get_reduced_evaluation_info(x) for x in evals]
        return eval_infos_reduced
        #raise NotImplementedError("get_available_brain_runs method not implemented")

    def get_reduced_evaluation_info(self, eval_key):
        info = self.dataset.get_evaluation_info(eval_key)
        cfg = info.config

        #CONFIG_KEYS_KEEP = ['method', 'cls', 'pred_field', 'gt_field', 'iou']
        CONFIG_KEYS_KEEP = ['method', 'pred_field', 'gt_field', 'iou']
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
        #prefix += ', '.join([x['key'] for x in available_evals])
        #prefix += '\n\n'
        body = self.generate_examples_prompt(query, available_evals)
        return (prefix + body).replace('{', '(').replace('}', ')')
    
    def generate_examples_prompt(self, query, available_evals):
        examples = self.get_examples()

        # remove single quotes from dict str to match examples template
        available_evals_str = str(available_evals).replace("'","")

        return FewShotPromptTemplate(
            examples=examples,
            example_prompt=EVAL_EXAMPLE_PROMPT,
            prefix=EVAL_PROMPT_PREFIX,
            suffix=EVAL_PROMPT_SUFFIX,
            input_variables=EVAL_PROMPT_INPUTS,
            example_separator="\n",
        ).format(
            query=query, 
            available_evals=available_evals_str, #            run_type=self.run_type
            )
    
    def get_examples(self):
        with open(self.examples_file, "r") as f:
            df = pd.read_csv(f)
        return df.to_dict('records') 
        '''
        examples = []

        for _, row in df.iterrows():
            example = {
                "query": row.query,
                "available_evaluations": row.available_evaluations,
                "selected_evaluation": row.selected_evaluation
            }
            examples.append(example)
        return examples
        '''
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
            prompt = '<single available evaluation>'
            resp = evals[0]['key']
        else:
            _, prompt, resp = self.llm_select_evaluation(query, evals=evals)
        
        return resp, prompt, evals

    
class DefaultEvaluationSelector(EvaluationSelector):
     def __init__(self, dataset):
        super().__init__(dataset)

def select_evaluation_run(dataset, query):
    selector = DefaultEvaluationSelector(dataset)
    return selector.select_evaluation_run(query)


class EvaluationYesNo:
    """Class to determine whether an evaluation run is necessary for a given query and dataset"""

    EXAMPLES_FILE = 'examples/fiftyone_evaluation_yn.csv'
    PREFIX_FILE = 'prompts/evaluation_yesno_prefix.txt'
    EXAMPLE_TEMPLATE = """
        Query: {query}
        Evaluation run used: {yesno}\n
        """
    EXAMPLE_TEMPLATE_INPUTS = ['query', 'yesno']

    def get_examples(self):
        df = pd.read_csv(self.EXAMPLES_FILE)
        return df.to_dict('records')

    def get_prompt_prefix(self):
        with open(self.PREFIX_FILE,"r") as f:
            prefix = f.read()
        return prefix
    
    def get_single_example_template(self):
        return PromptTemplate(
            input_variables=self.EXAMPLE_TEMPLATE_INPUTS,
            template=self.EXAMPLE_TEMPLATE,
        )

    def generate_prompt(self, query):
        examples = self.get_examples()
        example_template = self.get_single_example_template()
        prefix = self.get_prompt_prefix()
        
        prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_template,
            prefix=prefix,
            suffix="Query: {query}\nEvaluation run used:",
            input_variables=["query"],
            example_separator="\n",
        )

        return prompt.format(query = query)

    def generate_evaluation_yesno(self,query):
        prompt = self.generate_prompt(query)
        resp = llm.call_as_llm(prompt).strip()
        return resp, prompt






















