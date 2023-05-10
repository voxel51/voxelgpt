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
    all_text = ""
    for element in text[:-1]:
        all_text += element + "."
    all_text += text[-1]
    code = 'dataset.' + all_text
    print(code)
    try:
        view = eval(code)
    except:
        print("Bad View.")
        view = dataset.view()
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

def get_categories(view):
    categories = view.values("ground_truth.detections.label")
    flat_list = [item for sublist in categories for item in sublist]
    categories = list(set(flat_list))
    return categories

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
        elif get_categories(gpt_response) != get_categories(ground_truth):
            return False
        else:
            return True

    def test_hardness(self):
        prompt = "Find the 23 most difficult images in my dataset"
        dataset = foz.load_zoo_dataset("quickstart")
        compute_hardness_for_test(dataset)
        expected_view = create_view_from_stages(['sort_by("hardness",reverse=True)', 'limit(23)'], dataset)
        gpt_response = get_gpt_view_text(dataset, prompt)
        if type(gpt_response) == list:
            view = create_view_from_stages(gpt_response, dataset)
            assert self.EvaluateResults(expected_view, view)
        else:
            print("Response failed. Returned " + str(gpt_response))
            assert False
        
    def test_uniqueness(self):
        prompt = "Find the 19 most distinct samples in my dataset"
        dataset = foz.load_zoo_dataset("quickstart")
        fob.compute_uniqueness(dataset)
        expected_view = create_view_from_stages(['sort_by("uniqueness",reverse=True)', 'limit(19)'], dataset)
        gpt_response = get_gpt_view_text(dataset, prompt)
        if type(gpt_response) == list:
            view = create_view_from_stages(gpt_response, dataset)
            assert self.EvaluateResults(expected_view, view)
        else:
            print("Response failed. Returned " + str(gpt_response))
            assert False
        
    def test_mistakenness(self):
        prompt = "find my 8 biggest flaws"
        dataset = foz.load_zoo_dataset("quickstart")
        fob.compute_mistakenness(dataset, "predictions", label_field="ground_truth")
        expected_view = create_view_from_stages(['sort_by("mistakenness",reverse=True)', 'limit(8)'], dataset)
        gpt_response = get_gpt_view_text(dataset, prompt)
        if type(gpt_response) == list:
            view = create_view_from_stages(gpt_response, dataset)
            assert self.EvaluateResults(expected_view, view)
        else:
            print("Response failed. Returned " + str(gpt_response))
            assert False
        
    def test_similarity(self):
        prompt = "Find the 19 most similar samples to a lion in my dataset"
        dataset = foz.load_zoo_dataset("quickstart")
        fob.compute_similarity(dataset, model="clip-vit-base32-torch", brain_key="test_sim")
        expected_view = create_view_from_stages(['sort_by_similarity("lion", k=19)'], dataset)
        gpt_response = get_gpt_view_text(dataset, prompt)
        if type(gpt_response) == list:
            view = create_view_from_stages(gpt_response, dataset)
            assert self.EvaluateResults(expected_view, view)
        else:
            print("Response failed. Returned " + str(gpt_response))
            assert False
            
    def test_label(self):
        prompt = "Only show labels containing the letter m"
        dataset = foz.load_zoo_dataset("quickstart")
        expected_view = create_view_from_stages(['filter_labels("ground_truth", F("label").contains_str("m"))'], dataset)
        gpt_response = get_gpt_view_text(dataset, prompt)
        if type(gpt_response) == list:
            view = create_view_from_stages(gpt_response, dataset)
            assert self.EvaluateResults(expected_view, view)
        else:
            print("Response failed. Returned " + str(gpt_response))
            assert False