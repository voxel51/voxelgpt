# To Run: pytest -q tests/test_simple_functions.py
# See README for full details

import fiftyone as fo
from fiftyone import ViewField as F
import fiftyone.zoo as foz
import fiftyone.brain as fob
import random
import numpy as np
import sys
sys.path.append("/Users/leila.kaneda/gpt-view-stages") #Replace with the path to your folder
from gpt_view_generator import get_gpt_view_text

def create_view_from_stages(text, dataset):
    view = dataset.view()
    code = 'dataset.' + text.strip()[1:-1]
    view = eval(code)
    return view

def compute_hardness_for_test(dataset):
    # Create some fake classifications to make a hardness brain run,
    classes = ["sheep", "cat", "dog", "moose"]
    logits = np.random.normal(size = 4)
    logits /= logits.sum()
    for sample in dataset:
        sample["my_classifications"] = fo.Classification(label=random.choice(classes), logits=logits, confidence=random.random())
        sample.save()
    
    fob.compute_hardness(dataset, label_field="my_classifications")
    fob.compute_hardness(dataset, label_field="my_classifications", hardness_field="test_hardness")

class TestClassSimpleFunctions:
    def EvaluateResults(self, ground_truth, gpt_response):
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

    def test_hardness(self):
        prompt = "Find the 23 most difficult images in my dataset"
        dataset = foz.load_zoo_dataset("quickstart")
        compute_hardness_for_test(dataset)
        #expected_view = create_view_from_stages("[exclude_by('my_field', ['a', 'b', 'e', '1'])]", dataset)
        gpt_response = get_gpt_view_text(dataset, prompt)
        view = create_view_from_stages(gpt_response, dataset)
        #assert self.EvaluateResults(expected_view, view)
        print("the end")
        
    def test_uniqueness(self):
        prompt = "Find the 19 most distinct samples in my dataset"
        dataset = foz.load_zoo_dataset("quickstart")
        fob.compute_uniqueness(dataset)
        gpt_response = get_gpt_view_text(dataset, prompt)
        print(gpt_response)
        print("the end")
        
    def test_mistakenness(self):
        prompt = "Find my 8 biggest flaws"
        #prompt = "Find the 8 samples with the most mistakes"
        dataset = foz.load_zoo_dataset("quickstart")
        fob.compute_mistakenness(dataset, "predictions", label_field="ground_truth")
        gpt_response = get_gpt_view_text(dataset, prompt)
        print(gpt_response)
        print("the end")
        
    def test_similarity(self):
        prompt = "Find the 19 most similar samples to a cat in my dataset"
        dataset = foz.load_zoo_dataset("quickstart")
        fob.compute_similarity(dataset, model="clip-vit-base32-torch", brain_key="test_sim")
        gpt_response = get_gpt_view_text(dataset, prompt)
        print(gpt_response)
        print("the end")