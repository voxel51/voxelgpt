# To Run: pytest -q tests/test_prompts.py
# See README for full details

import fiftyone as fo
from fiftyone import ViewField as F
import fiftyone.zoo as foz
import sys
sys.path.append("/Users/leila.kaneda/gpt-view-stages/") #Replace with the path to your folder
from gpt_view_generator import get_gpt_view_text

class TestClassViewStages:
    def MockDataset(self, test_name):
        dataset = fo.Dataset("test_dataset_" + str(test_name))
        return dataset
        
    def EvaluateResults(self, gpt_response, ground_truth):
        # Option 1: Check if the dictionary representations are equal.
        # if gpt_response.to_dict() == ground_truth.to_dict():
        #     return True
        # else:
        #     return False
        
        # Option 2: Check that the sample counts, ids, filepaths, labels, and fields are the same
        if gpt_response.stats()['samples_count'] != ground_truth.stats()['samples_count']:
            return False
        elif sorted(gpt_response.values("id")) != sorted(ground_truth.values("id")):
            return False
        elif sorted(gpt_response.values("filepath")) != sorted(ground_truth.values("filepath")):
            return False
        elif gpt_response.values("ground_truth.detections.label") != ground_truth.values("ground_truth.detections.label"):
            return False
        elif gpt_response.get_field_schema() != ground_truth.get_field_schema():
            return False
        else:
            return True

    def test_create_dataset(self):
        ds = self.MockDataset("create_dataset")
        assert fo.dataset_exists("test_dataset_create_dataset")
        assert ds.name == "test_dataset_create_dataset"
        
    def test_query_1(self):
        prompt = "Show me images with no detections"
        ds = foz.load_zoo_dataset("coco-2017", split="validation")
        gt = ds.match(F('ground_truth.detections').length() == 0)
        gpt_response = get_gpt_view_text(ds, prompt)
        assert self.EvaluateResults(gpt_response, gt)