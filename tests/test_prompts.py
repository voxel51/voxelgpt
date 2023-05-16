"""
Prompt tests.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os
import sys

import re

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpt_view_generator import ask_gpt_generator


def get_gpt_view_text(dataset, query):
    response = None
    for response in ask_gpt_generator(dataset, query, raw=True):
        pass

    return response



def remove_whitespace(stage_str):
    return re.sub(
        r'\s+', lambda m: ' ' if len(m.group(0)) == 1 else '',
        stage_str
        )


def split_into_stages(stages_text):
    with open("view_stages_list.txt", "r") as f:
        view_stages = f.read().splitlines()
    pattern = ','+'|,'.join(view_stages)[:-1]

    st = stages_text[1:-1].replace(', ', ',').replace('\n', '')
    st = st.replace('\r', '').replace('\'', "\"")
    x = re.finditer(pattern, st)

    stages = []
    spans = []
    for match in x:
        spans.append(match.span())

    spans = spans[::-1]
    for i, span in enumerate(spans):
        if i == 0:
            stages.append(st[span[0]+1:])
        else:
            stages.append(st[span[0]+1:spans[i-1][0]])
    if len(stages) != 0:
        stages.append(st[:spans[-1][0]])
    else:
        stages.append(st)

    stages = stages[::-1]
    stages = [remove_whitespace(stage) for stage in stages]
    return stages


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


class TestClassViewStages:
    # def MockDataset(self, test_name):
    #     dataset = fo.Dataset("test_dataset_" + str(test_name))
    #     return dataset

    def EvaluateResults(self, ground_truth, gpt_response):
        assert gpt_response.stats()['samples_count'] == ground_truth.stats()['samples_count']
        assert sorted(gpt_response.values("id")) == sorted(ground_truth.values("id"))
        assert sorted(gpt_response.values("filepath")) == sorted(ground_truth.values("filepath"))
        assert gpt_response.values("ground_truth.detections.label") == ground_truth.values("ground_truth.detections.label")
        assert gpt_response.get_field_schema() == ground_truth.get_field_schema()
        
    def test_query1(self):
        prompt = "Create a view excluding samples whose `my_field` field have values in ['a', 'b', 'e', '1']"
        dataset = foz.load_zoo_dataset("quickstart")
        expected_view = create_view_from_stages(
            "[exclude_by('my_field', ['a', 'b', 'e', '1'])]", dataset
        )
        gpt_response = get_gpt_view_text(dataset, prompt)
        print(gpt_response)
        view = create_view_from_stages(gpt_response, dataset)
        assert self.EvaluateResults(expected_view, view)

    '''
    def test_query2(self):
        prompt = (
            "remove samples with 1, 3, 5, 7, or 9 in 'num_predictions' field"
        )
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[exclude_by('num_predictions', [1, 3, 5, 7, 9])]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages, dataset)
        gpt_response = get_gpt_view_text(dataset, prompt)
        print(gpt_response)
        view = create_view_from_stages(gpt_response, dataset)
        self.EvaluateResults(expected_view,view)
    '''

    def test_100_random_samples_of_dogs_with_people(self):
        prompt = "100 random samples of dogs with people"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[match(F('ground_truth.detections.label').contains(['dog', 'person'], all=True)), take(100)]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages, dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    '''
    def test_hardest_sample_of_a_random_sampling_of_69_samples(self):
        prompt = "Hardest sample of a random sampling of 69 samples"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[take(69), sort_by('hardness', reverse=True), limit(1)]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages, dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)
    
    def test_show_me_the_clip_trajectories_of_my_detections_wit(self):
        prompt = "Show me the clip trajectories of my detections with no people or road signs"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[exclude_by('detections.label',['person', 'road sign']), to_trajectories('frames.detections')]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)
    
    def test_show_me_crowded_videos_in_the_daytime(self):
        prompt = "Show me crowded videos in the daytime"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[match_frames(F('detections.detections').length() > 100), match(F('detections.detections.label.timeofday').is_subset(['daytime']))]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_show_me_11_random_samples_of_people_playing_sports(self):
        prompt = "show me 11 random samples of people playing sports"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[sort_by_similarity('people playing sports', k = 15, brain_key = 'qdrant'), take(11)]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_show_me_the_hardest_samples_in_mumbai(self):
        prompt = "Show me the hardest samples in Mumbai"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[sort_by('hardness', reverse=True), geo_near([72.8777, 19.0760])]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_show_me_all_confirmed_blocked_exit_violations_with(self):
        prompt = "Show me all confirmed blocked exit violations within Amazon Headquarters in Seattle, WA"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[match(F('eval')==True).filter_labels('blocked_exit'), geo_within([-122.340524,47.617944], [-122.336709,47.61571], [-122.338324,47.614524], [-122.342042,47.616716])]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_find_me_images_nearby_and_east_from_ann_arbor(self):
        prompt = "Find me images nearby and east from Ann Arbor"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[match(((F('geolocation_field.point.coordinates')[0] + 83.732124 - 1).abs() < 1) & ((F('geolocation_field.point.coordinates')[1] - 42.279594).abs() < 1))]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_show_me_houses_in_ann_arbor_mi(self):
        prompt = "Show me houses in Ann Arbor, MI"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[match(F('ground_truth.detections.label').is_subset(['house'])), geo_near([83.7430, 42.2808], max_distance=80467.2)]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_find_all_samples_within_100km_of_paris_either_pt_n(self):
        prompt = "Find all samples within 100km of Paris (either pt named paris, or latlong)"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[geo_near([2.3522, 48.8566], max_distance=100000)]", datase
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_29_sunny_in_philadelphia_samples(self):
        prompt = "29 sunny in philadelphia samples"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[match(F('weather').label == 'sunny'), geo_near([75.1652, 39.9526]), take(29)]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)
    '''

    def test_give_me_all_videos_that_are_longer_than_5_seconds(self):
        prompt = "Give me all videos that are longer than 5 seconds"
        dataset = foz.load_zoo_dataset("quickstart-video")
        dataset.compute_metadata()
        stages = "[match(F('metadata.duration') > 5)]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    '''
    def test_show_me_80_hardest_samples_with_incorrect_predicti(self):
        prompt = "Show me 80 hardest samples with incorrect predictions having confidence above 0.82"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[match(F('eval')==False).filter_labels('prediction',F('confidence')>0.82), sort_by('hardness', reverse=True), limit(80)]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)
    '''

    def test_show_me_images_with_no_detections(self):
        prompt = "Show me images with no detections"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[match(F('detections.detections').length() == 0)]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    '''
    def test_missed_predictions_with_no_annotation_mistakes(self):
        prompt = "Missed predictions with no annotation mistakes"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[match(F('eval')==False), exclude_labels(tags='annotation_mistake')]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_missed_predictions_from_torch_model_with_no_annota(self):
        prompt = (
            "Missed predictions from torch model with no annotation mistakes"
        )
        dataset = foz.load_zoo_dataset("quickstart")
        expected_view = create_view_from_stages(
            "[match(F('eval')==False).filter_labels('torch'), exclude_labels(tags='annotation_mistake')]",
            dataset,
        )
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        assert self.EvaluateResults(expected_view, gpt_view)
    
    def test_3_least_unique_black_and_white_images(self):
        prompt = "3 least unique black and white images"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[match(F('metadata.num_channels') = 1), sort_by('uniqueness', reverse=False), limit(3)]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_samples_containing_eyes_without_glasses(self):
        prompt = "Samples containing eyes without glasses"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[match(F('positive_labels.detections.label').contains('human eye') & F('negative_label.detections.label').contains(['glasses', 'sunglasses', 'monacle'])]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_45_nighttime_samples_with_most_mistakes_this_year(self):
        prompt = "45 nighttime samples with most mistakes this year"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[match(F('detections.detections.label.timeofday').is_subset(['nighttime'])), match(F('timetaken').year() == 2023), sort_by('mistakenness', reverse=True), limit(45)]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_most_unique_images_from_2020_similar_to_image_101(self):
        prompt = "Most unique images from 2020 similar to image 101."
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[match(F('date').year() == 2020), sort_by_similarity(dataset.limit[101:].first().id), sort_by('uniqueness', reverse=True)]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_video_clips_with_drones_flying_during_the_golden_h(self):
        prompt = "Video clips with drones flying during the golden hour"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[match(F('detections.detections.label.timeofday').is_subset(['dawn/dusk'])), filter_labels('events', F('label') == 'drone'), to_clips('events')]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_trajectories_containing_swimming_fish(self):
        prompt = "Trajectories containing swimming fish"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[filter_labels('frames.detections', F('label') == 'fish'), to_trajectories('frames.detections')]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_top_10_mistakes_from_cvat_annotation_and_labelstud(self):
        prompt = "Top 10 mistakes from CVAT annotation and LabelStudio annotation respectively"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[select_labels(tags=['cvat', 'annotation_mistake']), limit(10), concat(select_labels(tags=['label_studio', 'annotation_mistake']), limit(10))]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_show_me_lots_of_small_objects_from_my_model_predic(self):
        prompt = "Show me lots of small objects from my model predictions"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[filter_labels('rcs_5_20230417', (F('bounding_box')[2] * F('bounding_box')[3]) < 0.1), match(F('rcs_5_20230417.detections').length() > 10)]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_what_are_the_5_most_likley_object_detection_mistak(self):
        prompt = "What are the 5 most likley object detection mistakes in my dataset?"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[filter_labels('final_yolov8', (F('march_1_eval') == 'fp') & (F('confidence') > 0.8)), sort_by(F('final_yolov8.detections').reduce(VALUE.append(F('confidence')), init_val=[]).max()), limit(5)]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_remove_all_non_person_or_car_objects_from_my_model(self):
        prompt = "Remove all non-person or car objects from my model predictions but keep all samples"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[filter_labels('instances', F('label').is_in(['person', 'Car'], only_matches=False)]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_which_airplanes_are_occluded_for_longer_than_20_fr(self):
        prompt = "Which airplanes are occluded for longer than 20 frames?"
        dataset = foz.load_zoo_dataset("quickstart-video")
        stages = "[filter_labels('frames.close_poly_segs', (F('label') == 'airplane') & (F('occluded') == True), trajectories=True), to_trajectories('frames.close_poly_segs'), match(F('frames.close_poly_segs').filter(F('occluded') == True)).length() > 20)]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_give_me_the_100_most_unique_images_with_a_keypoint(self):
        prompt = "Give me the 100 most unique images with a keypoint that has an FTON score of at least 51"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[filter_keypoints('driver_pose', F('FTON') >= 51), sort_by('uniqueness', reverse=True), limit(100)]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_round_the_chs_score_of_my_scenes_to_one_decimal_pl(self):
        prompt = "Round the CHS score of my scenes to one decimal place"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[set_field('scene', F('scene.classifications').map(F('chs_score').round(place=1)))]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_filter_for_all_tags_that_include_annotation_mistak(self):
        prompt = "Filter for all tags that include 'annotation_mistake', 'june_task_5', 'roi_issue'. Case independent"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[set_field('tags', F('tags').map(F().lower())), match_tags(['annotation_mistake', 'june_task_5', 'roi_issue'])]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_which_patches_from_my_frames_show_potholes_in_a_cr(self):
        prompt = "Which patches from my frames show potholes in a crowded scene of more than 20 objects?"
        dataset = foz.load_zoo_dataset("quickstart-video")
        stages = "[to_frames(), match(F('test_detections.detections').length() > 20), filter_labels('test_detections', F('label') == 'Pot hole'),  to_patches('test_detections')]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_update_my_crowding_field_based_on_the_number_of_pe(self):
        prompt = "Update my crowding field based on the number of people in the image given the following: {0 people: empty, 1 person: solo, 2 people: group, 10 people: gathering, 15+ people: crowd}"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[set_field('crowding', F('person_detector.detections').length().switch({(F()==0): 'empty', (F() == 1)): 'solo', ((F() > 2) & (F() <= 10)): 'group', ((F() > 10) & (F() <= 15)): 'gathering', (F() > 15): 'crowd' }))]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_show_all_of_my_model_predictions_from_models_1_3_i(self):
        prompt = (
            "Show all of my model predictions from models 1-3 in one field"
        )
        dataset = foz.load_zoo_dataset("quickstart")
        expected_view = create_view_from_stages(
            "[set_field('model_1', F('detections').extend(F('$model_2.detections')).extend(F('$model_3.detections')))]",
            dataset,
        )
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        assert self.EvaluateResults(expected_view, gpt_view)

    def test_show_me_samples_with_lots_of_small_objects_from_my(self):
        prompt = "Show me samples with lots of small objects from my model predictions"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[match(F('rcs_5_20230417.detections').filter(F('bounding_box')[2] * F('bounding_box')[3]) < 0.1).length() > 10)]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_when_did_i_detect_objects_best_at_night_or_dusk(self):
        prompt = "When did I detect objects best at night or dusk?"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[match(F('timeofday').is_in(['night', 'dusk'])), to_evaluation_patches('eval', other_fields=['timeofday'])]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_which_segmentations_did_eric_annotate_in_january(self):
        prompt = "Which segmentations did Eric annotate in January?"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[filter_labels('segmentations', (F('annotator') == 'Eric') & F('annotation_date').month() == 1))]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_show_me_all_patches_with_issues(self):
        prompt = "Show me all patches with issues"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[to_patches('gold_standard'), match_labels(tags=['box_issue', 'class_issue'])]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_show_me_the_10_samples_with_classifications_most_s(self):
        prompt = "Show me the 10 samples with classifications most similar to the most incorrect prediction"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[sort_by_similarity(dataset.match(F('eval') == False).sort_by('predictions.confidence').first().id), limit(10)]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_show_me_all_samples_that_are_visually_at_night_but(self):
        prompt = "Show me all samples that are visually at night, but not annotated as night"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[match(F('timeofday') != 'night'), sort_by_similarity('image taken at night')]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_what_are_the_100_most_unique_samples_that_i_havent(self):
        prompt = (
            "What are the 100 most unique samples that I haven't trained on"
        )
        dataset = foz.load_zoo_dataset("quickstart")
        expected_view = create_view_from_stages(
            "[match_tags('train', bool=False), sort('uniqueness'), limit(100)]",
            dataset,
        )
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        assert self.EvaluateResults(expected_view, gpt_view)

    def test_select_all_video_frames_that_do_not_contain_precis(self):
        prompt = (
            "Select all video frames that do not contain precisely two objects"
        )
        dataset = foz.load_zoo_dataset("quickstart-video")
        expected_view = create_view_from_stages(
            "[match_frames(F('detections.detections').length() !=2)]", dataset
        )
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        assert self.EvaluateResults(expected_view, gpt_view)

    def test_find_all_video_frames_that_contain_an_object_with_(self):
        prompt = "Find all video frames that contain an object with aspect ratio greater than 2 or less than 0.5"
        dataset = foz.load_zoo_dataset("quickstart-video")
        stages = "[filter_labels('frames.detections', (F('bounding_box')[2]/F('bounding_box')[3] > 2) |  (F('bounding_box')[2]/F('bounding_box')[3] <0.5) )]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_find_all_samples_where_the_filepath_contains_montr(self):
        prompt = "Find all samples where the filepath contains ‘montreal’ and ‘trial2’"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[match(F('filepath').contains_str(['montreal','trial2']))]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        print(gpt_view_stages)
        gpt_view = create_view_from_stages(gpttages, dataset)
        assert self.EvaluateResults(expected_view, gpt_view)

    def test_which_samples_are_in_the_task_16_directory(self):
        prompt = "Which samples are in the 'task_16' directory?"
        dataset = foz.load_zoo_dataset("quickstart")
        expected_view = create_view_from_stages(
            "[match(F('filepath').contains_str('task_16'))]", dataset
        )
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        print(gpt_view_stages)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        assert self.EvaluateResults(expected_view, gpt_view)

    def test_create_a_clips_view_for_all_bouts_of_attack_or_cha(self):
        prompt = "Create a clips view for all bouts of ‘attack’ or ‘chase’"
        dataset = foz.load_zoo_dataset("quickstart-video")
        stages = "[filter_labels('bouts', F('label').is_in(['attack','label]')), to_clips('bouts')]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_find_all_images_from_zip_codes_07920_07924_and_070(self):
        prompt = "Find all images from zip codes 07920, 07924, and 07059 that were sunny"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[F('zip_code').is_in(['07920','07924','07059']), F('weather')=='sunny')]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_find_all_video_samples_tagged_as_intruder_that_do_(self):
        prompt = "Find all (video) samples tagged as ‘Intruder’ that do not contain a vehicle"
        dataset = foz.load_zoo_dataset("quickstart-video")
        stages = "[match_tags('intruder'), match(~F('frames').filter(F('obj_dets.detections').filter(F('label') == 'vehicle').length() > 1)).length() > 1))]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_find_all_samples_that_are_not_in_a_group_of_at_lea(self):
        prompt = "Find all samples that are not in a group of at least 5 when grouped by 'status'"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[group_by('status',match_expr=F().length()<5)] errors on displaying view"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_find_all_incorrect_detections_where_model_date_is_(self):
        prompt = "Find all incorrect detections where model_date is earlier than June 2020"
        dataset = foz.load_zoo_dataset("quickstart")
        expected_view = create_view_from_stages(
            "[filter_labels('predictions',F('eval00')!='tp' & F('model_date')<datetime.datetime(2020,6))]",
            dataset,
        )
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        assert self.EvaluateResults(expected_view, gpt_view)

    def test_restrict_the_labels_to_just_the_keypoints(self):
        prompt = "Restrict the labels to just the keypoints"
        dataset = foz.load_zoo_dataset("quickstart")
        expected_view = create_view_from_stages(
            "[select_fields('landmarks')]", dataset
        )
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        assert self.EvaluateResults(expected_view, gpt_view)

    def test_restrict_the_labels_to_just_the_keypoints_video_da(self):
        prompt = "Restrict the labels to just the keypoints (video dataset)"
        dataset = foz.load_zoo_dataset("quickstart-video")
        expected_view = create_view_from_stages(
            "[select_fields('frames.landmarks')]", dataset
        )
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        assert self.EvaluateResults(expected_view, gpt_view)

    def test_find_all_images_tagged_as_annotationerror_but_not_(self):
        prompt = "Find all images tagged as AnnotationError but not Reviewer2"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[match_tags('AnnotationError'), match_tags('Reviewer2',bool=False)]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_find_all_labels_tagged_as_annotationerror_but_not_(self):
        prompt = "Find all labels tagged as AnnotationError but not Reviewer2"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[select_labels(tags='AnnotationError'), exclude_labels(tags='Reviewer2')]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_sort_the_images_in_decreasing_order_of_the_hazard_(self):
        prompt = (
            "Sort the images in decreasing order of the ‘hazard’ statistic"
        )
        dataset = foz.load_zoo_dataset("quickstart")
        expected_view = create_view_from_stages(
            "[sort_by('hazard',reverse=True)]", dataset
        )
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        assert self.EvaluateResults(expected_view, gpt_view)

    def test_find_all_images_that_do_not_have_a_fall_hazard_sta(self):
        prompt = "Find all images that do not have a “fall_hazard” statistic computed"
        dataset = foz.load_zoo_dataset("quickstart")
        expected_view = create_view_from_stages(
            "[match(~F('fall_hazard'))]", dataset
        )
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        assert self.EvaluateResults(expected_view, gpt_view)

    def test_find_all_frames_captured_within_one_day_of_june_4_(self):
        prompt = "Find all frames captured within one day of June 4, 2023"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[match( abs(F('timestamp') - datetime.datetime(2023,6,4)) < datetime.timedelta(days=1) )]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_exclude_all_samples_where_the_status_is_released(self):
        prompt = "Exclude all samples where the status is ‘released’"
        dataset = foz.load_zoo_dataset("quickstart")
        expected_view = create_view_from_stages(
            "[match( F('status')!='released' )]", dataset
        )
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        assert self.EvaluateResults(expected_view, gpt_view)

    def test_isolate_object_index_5_in_a_trajectories_view(self):
        prompt = "Isolate object index 5 in a trajectories view"
        dataset = foz.load_zoo_dataset("quickstart-video")
        stages = "[filter_labels('frames.detections',F('index')==5), to_trajectories('frames.detections')]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_create_a_frames_view_for_every_10th_frame_in_each_(self):
        prompt = "Create a frames view for every 10th frame in each video"
        dataset = foz.load_zoo_dataset("quickstart-video")
        expected_view = create_view_from_stages(
            "[match_frames(F('frame_number') % 10 == 0)]", dataset
        )
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        assert self.EvaluateResults(expected_view, gpt_view)

    def test_return_a_view_with_every_10th_sample(self):
        prompt = "Return a view with every 10th sample"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[match(F('filepath').is_in(dataset.values('filepath')[::10]))]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_show_any_sample_where_the_model_made_a_mistake_in_(self):
        prompt = "Show any sample where the model made a mistake in prediction"
        dataset = foz.load_zoo_dataset("quickstart")
        expected_view = create_view_from_stages(
            "[match( (F('eval00_fp')>0) | (F('eval00_fn')>0) )]", dataset
        )
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        assert self.EvaluateResults(expected_view, gpt_view)

    def test_find_all_missed_detections_of_license_plates(self):
        prompt = "Find all missed detections of license plates"
        dataset = foz.load_zoo_dataset("quickstart")
        stages = "[filter_labels('ground_truth',F('eval00')=='fn' & F('label')=='license plate'), to_evaluation_patches('eval00')]"
        stages = split_into_stages(stages)
        expected_view = create_view_from_stages(stages,dataset)
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        self.EvaluateResults(expected_view, gpt_view)

    def test_which_samples_have_a_store_id(self):
        prompt = "Which samples have a store id?"
        dataset = foz.load_zoo_dataset("quickstart")
        expected_view = create_view_from_stages(
            "[exists('store_id')]", dataset
        )
        gpt_view_stages = get_gpt_view_text(dataset, prompt)
        gpt_view = create_view_from_stages(gpt_view_stages, dataset)
        assert self.EvaluateResults(expected_view, gpt_view)
    '''
