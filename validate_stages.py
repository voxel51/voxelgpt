import re

def validate_view_stage(stage_str):
    stage, args = parse_view_stage(stage_str)
    print(stage)
    print(args)
    validator = StageValidatorFactory().get(stage)
    validator.validate(args)

def parse_view_stage(stage_str):
    """parse a view stage string into a dictionary"""

    stage = stage_str.split('(')[0].strip()
    args = '('.join(stage_str.split('(')[1:])[:-1]
    args = [a.strip() for a in args.split(', ')]

    positional_args = []
    named_args = {}

    # Loop through each argument
    for arg in args:
        # Check if the argument is a named argument
        if "=" in arg:
            # Split the argument on the first occurrence of "="
            key, value = arg.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key.replace('.', '').isalpha():
                named_args[key] = value
            else:
                positional_args.append(arg)
        else:
            positional_args.append(arg)

    return (stage, positional_args, named_args)

###############################################################################

class TypeValidatorFactory(object):
    """factory for creating type validators"""
    def __init__(self, arg):
        self._validators = {
            "int": IntValidator(),
            "bool": BoolValidator(),
        }

    def get(self, name):
        """get a type validator"""
        return self._validators[name]

class TypeValidator():
    """base class for validating a type"""
    def __init__(self):
        pass
    
    def validate(self, arg):
        return True

class IntValidator():
    "validator for arguments of type int"
    def __init__(self):
        super.__init__(self)

    def validate(self, arg):
        if not arg.isdigit():
            raise ValueError("must be an integer")

class BoolValidator():
    "validator for arguments of type bool"
    def __init__(self):
        super.__init__(self)

    def validate(self, arg):
        if arg not in ["True", "False"]:
            raise ValueError("must be a bool")


NAMED_ARG_TYPES = {
    "seed": "int",
    "k": "int",
    "tol": "int",
    "min_len": "int",
    "fps": "int",
    "max_fps": "int",
    "size": "int",
    "min_size": "int",
    "max_size": "int",
    "bool": "bool",
    "all": "bool",
    "omit_empty": "bool",
    "only_matches": "bool",
    "ordered": "bool",
    "reverse": "bool",
    "keep_label_list": "bool",
    "trajectories": "bool",
    "sample_frames": "bool",
    "sparse": "bool",
    "force_samples": "bool",
    "skip_failures": "bool",
    "verbose": "bool",
    "output_dir": "str",
    "rel_dir": "str",
    "frames_patt": "str"
}

###############################################################################

class StageValidatorFactory():
    """factory for creating stage validators"""
    def __init__(self):
        self._validators = {
            "exclude": ExcludeViewStageValidator(),
            "exclude_by": ExcludeByViewStageValidator(),
            "exclude_fields": ExcludeFieldsViewStageValidator(),
            "exclude_frames": ExcludeFramesViewStageValidator(),
            "exclude_labels": ExcludeLabelsViewStageValidator(),
            "exists": ExistsViewStageValidator(),
            "filter_field": FilterFieldViewStageValidator(),
            "filter_labels": FilterLabelsViewStageValidator(),
            "filter_keypoints": FilterKeypointsViewStageValidator(),
            "geo_near": GeoNearViewStageValidator(),
            "geo_within": GeoWithinViewStageValidator(),
            "limit": LimitViewStageValidator(),
            "limit_labels": LimitLabelsViewStageValidator(),
            "map_labels": MapLabelsViewStageValidator(),
            "set_field": SetFieldViewStageValidator(),
            "match": MatchViewStageValidator(),
            "match_frames": MatchFramesViewStageValidator(),
            "match_labels": MatchLabelsViewStageValidator(),
            "match_tags": MatchTagsViewStageValidator(),
            "select": SelectViewStageValidator(),
            "select_by": SelectByViewStageValidator(),
            "select_fields": SelectFieldsViewStageValidator(),
            "select_frames": SelectFramesViewStageValidator(),
            "select_labels": SelectLabelsViewStageValidator(),
            "shuffle": ShuffleViewStageValidator(),
            "skip": SkipViewStageValidator(),
            "sort_by": SortByViewStageValidator(),
            "sort_by_similarity": SortBySimilarityViewStageValidator(),
            "take": TakeViewStageValidator(),
            "to_patches": ToPatchesViewStageValidator(),
            "to_evaluation_patches": ToEvaluationPatchesViewStageValidator(),
            "to_clips": ToClipsViewStageValidator(),
            "to_trajectories": ToTrajectoriesViewStageValidator(),
            "to_frames": ToFramesViewStageValidator(),
        }

    def get(self, name):
        """get a stage validator"""
        return self._validators[name]
    
class StageValidator():
    """base class for validating a view stage"""
    def __init__(self):
        pass
    
    def validate_num_positional_args(self, positional_args):
        return len(positional_args) == self.num_positional_args

    def validate_named_arg_names(self, named_args):
        for name in list(named_args.keys()):
            if name not in self.named_args:
                raise ValueError(
                    f"invalid argument {name} passed to {self.stage_name} view stage" 
                    )
            
    def validate_named_arg_types(self, named_args):
        for name, val in named_args.items():
            if name in list(NAMED_ARG_TYPES.keys()):
                TypeValidatorFactory(NAMED_ARG_TYPES[name]).validate(val)

    def validate(self, stage_args):
        """validate a stage"""
        positional_args, named_args = stage_args
        self.validate_num_positional_args(positional_args)
        self.validate_named_arg_names(named_args)
        self.validate_specific_stage(stage_args)
    
    def validate_specific_stage(self, stage_args):
        return "Not implemented"
    


###########################################################################

class ExcludeViewStageValidator(StageValidator):
    """validator for the exclude view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'exclude'
        self.num_positional_args = 1
        self.named_args = []

    def validate_specific_stage(self, stage_args):
        """validate the exclude view stage"""
        return True

class ExcludeByViewStageValidator(StageValidator):
    """validator for the exclude_by view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'exclude_by'
        self.num_positional_args = 2
        self.named_args = []

    def validate_specific_stage(self, stage_args):
        """validate the exclude_by view stage"""
        return True

class ExcludeFieldsViewStageValidator(StageValidator):
    """validator for the exclude_fields view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'exclude_fields'
        self.num_positional_args = 1
        self.named_args = []

    def validate_specific_stage(self, stage_args):
        """validate the exclude_fields view stage"""
        return True
    
class ExcludeFramesViewStageValidator(StageValidator):
    """validator for the exclude_frames view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'exclude_frames'
        self.num_positional_args = 1
        self.named_args = ["omit_empty"]

    def validate_specific_stage(self, stage_args):
        """validate the exclude_frames view stage"""
        return True

class ExcludeLabelsViewStageValidator(StageValidator):
    """validator for the exclude_labels view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'exclude_labels'
        self.num_positional_args = 0
        self.named_args = ["labels", "ids", "tags", "fields", "omit_empty"]

    def validate_specific_stage(self, stage_args):
        """validate the exclude_labels view stage"""
        return True    
    
class ExistsViewStageValidator(StageValidator):
    """validator for the exists view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'exists'
        self.num_positional_args = 1
        self.named_args = ["bool"]

    def validate_specific_stage(self, stage_args):
        """validate the exists view stage"""
        return True
    
class FilterFieldViewStageValidator(StageValidator):
    """validator for the filter_field view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'filter_field'
        self.num_positional_args = 2
        self.named_args = ["only_matches"]

    def validate_specific_stage(self, stage_args):
        """validate the filter_field view stage"""
        return True

class FilterLabelsViewStageValidator(StageValidator):
    """validator for the filter_labels view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'filter_labels'
        self.num_positional_args = 2
        self.named_args = ["only_matches", "trajectories"]

    def validate_specific_stage(self, stage_args):
        """validate the filter_labels view stage"""
        return True
    
class FilterKeypointsViewStageValidator(StageValidator):
    """validator for the filter_keypoints view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'filter_keypoints'
        self.num_positional_args = 1
        self.named_args = ["filter", "labels", "only_matches"]

    def validate_specific_stage(self, stage_args):
        """validate the filter_keypoints view stage"""
        return True
    
class GeoNearViewStageValidator(StageValidator):
    """validator for the geo_near view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'geo_near'
        self.num_positional_args = 1
        self.named_args = ["location_field", "min_distance", "max_distance"]

    def validate_specific_stage(self, stage_args):
        """validate the geo_near view stage"""
        return True
    
class GeoWithinViewStageValidator(StageValidator):
    """validator for the geo_within view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'geo_within'
        self.num_positional_args = 1
        self.named_args = ["location_field", "strict"]

    def validate_specific_stage(self, stage_args):
        """validate the geo_within view stage"""
        return True
    
class LimitViewStageValidator(StageValidator):
    """validator for the limit view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'limit'
        self.num_positional_args = 1
        self.named_args = []

    def validate_specific_stage(self, stage_args):
        """validate the limit_labels view stage"""
        return True

    def validate_specific_stage(self, stage_args):
        """validate the limit view stage"""
        limit = stage_args[0][0]
        return IntValidator().validate(limit)
    
class LimitLabelsViewStageValidator(StageValidator):
    """validator for the limit_labels view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'limit_labels'
        self.num_positional_args = 2
        self.named_args = []

    def validate_specific_stage(self, stage_args):
        """validate the limit_labels view stage"""
        return True
    
class MapLabelsViewStageValidator(StageValidator):
    """validator for the map_labels view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'map_labels'
        self.num_positional_args = 2
        self.named_args = []

    def validate_specific_stage(self, stage_args):
        """validate the map_labels view stage"""
        return True

class SetFieldViewStageValidator(StageValidator):
    """validator for the set_field view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'set_field'
        self.num_positional_args = 2

    def validate_specific_stage(self, stage_args):
        """validate the set_field view stage"""
        return True
    
class MatchViewStageValidator(StageValidator):
    """validator for the match view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'match'
        self.num_positional_args = 1
        self.named_args = []

    def validate_specific_stage(self, stage_args):
        """validate the match view stage"""
        return True
    
class MatchFramesViewStageValidator(StageValidator):
    """validator for the match_frames view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'match_frames'
        self.num_positional_args = 1
        self.named_args = ["omit_empty"]

    def validate_specific_stage(self, stage_args):
        """validate the match_frames view stage"""
        return True
    
class MatchLabelsViewStageValidator(StageValidator):
    """validator for the match_labels view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'match_labels'
        self.num_positional_args = 0
        self.named_args = ["labels", "ids", "tags", "filter", "fields", "bool"]

    def validate_specific_stage(self, stage_args):
        """validate the match_labels view stage"""
        return True

class MatchTagsViewStageValidator(StageValidator):
    """validator for the match_tags view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'match_tags'
        self.num_positional_args = 1
        self.named_args = ["bool", "all"]

    def validate_specific_stage(self, stage_args):
        """validate the match_tags view stage"""
        return True
    
class SelectViewStageValidator(StageValidator):
    """validator for the select view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'select'
        self.num_positional_args = 1
        self.named_args = ["ordered"]

    def validate_specific_stage(self, stage_args):
        """validate the select view stage"""
        return True

class SelectByViewStageValidator(StageValidator):
    """validator for the select_by view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'select_by'
        self.num_positional_args = 2
        self.named_args = ["ordered"]

    def validate_specific_stage(self, stage_args):
        """validate the select_by view stage"""
        return True

class SelectFieldsViewStageValidator(StageValidator):
    """validator for the select_fields view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'select_fields'
        self.num_positional_args = 0
        self.named_args = ["field_names"]

    def validate_specific_stage(self, stage_args):
        """validate the select_fields view stage"""
        return True
    
class SelectFramesViewStageValidator(StageValidator):
    """validator for the select_frames view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'select_frames'
        self.num_positional_args = 1
        self.named_args = ["omit_empty"]

    def validate_specific_stage(self, stage_args):
        """validate the select_frames view stage"""
        return True
    
class SelectLabelsViewStageValidator(StageValidator):
    """validator for the select_labels view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'select_labels'
        self.num_positional_args = 0
        self.named_args = ["labels", "ids", "tags", "fields", "omit_empty"]

    def validate_specific_stage(self, stage_args):
        """validate the select_labels view stage"""
        return True

class ShuffleViewStageValidator(StageValidator):
    """validator for the shuffle view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'shuffle'
        self.num_positional_args = 0
        self.named_args = []

    def validate_specific_stage(self, stage_args):
        """validate the shuffle view stage"""
        return True
    
class SkipViewStageValidator(StageValidator):
    """validator for the skip view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'skip'
        self.num_positional_args = 1
        self.named_args = []

    def validate_specific_stage(self, stage_args):
        """validate the skip view stage"""
        skip = stage_args[0][0]
        return IntValidator().validate(skip)
    
class SortByViewStageValidator(StageValidator):
    """validator for the sort_by view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'sort_by'
        self.num_positional_args = 1
        self.named_args = ["reverse"]

    def validate_specific_stage(self, stage_args):
        """validate the sort_by view stage"""
        return True
    
class SortBySimilarityViewStageValidator(StageValidator):
    """validator for the sort_by_similarity view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'sort_by_similarity'
        self.num_positional_args = 1
        self.named_args = ["k", "reverse", "dist_field", "brain_key"]

    def validate_specific_stage(self, stage_args):
        """validate the sort_by_similarity view stage"""
        return True
    
class TakeViewStageValidator(StageValidator):
    """validator for the take view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'take'
        self.num_positional_args = 1
        self.named_args = ["seed"]

    def validate_specific_stage(self, stage_args):
        """validate the take view stage"""
        take = stage_args[0][0]
        return IntValidator().validate(take)

class ToPatchesViewStageValidator(StageValidator):
    """validator for the to_patches view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'to_patches'
        self.num_positional_args = 1
        self.named_args = ["other_fields", "keep_label_list"]

    def validate_specific_stage(self, stage_args):
        """validate the to_patches view stage"""
        return True

class ToEvaluationPatchesViewStageValidator(StageValidator):
    """validator for the to_evaluation_patches view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'to_evaluation_patches'
        self.num_positional_args = 1
        self.named_args = ["other_fields"]

    def validate_specific_stage(self, stage_args):
        """validate the to_evaluation_patches view stage"""
        return True
    
class ToClipsViewStageValidator(StageValidator):
    """validator for the to_clips view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'to_clips'
        self.num_positional_args = 1
        self.named_args = ["other_fields", "tol", "min_len", "trajectories"]

    def validate_specific_stage(self, stage_args):
        """validate the to_clips view stage"""
        return True
    
class ToTrajectoriesViewStageValidator(StageValidator):
    """validator for the to_trajectories view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'to_trajectories'
        self.num_positional_args = 1
        self.named_args = ["other_fields", "tol", "min_len", "trajectories"]

    def validate_specific_stage(self, stage_args):
        """validate the to_trajectories view stage"""
        return True
    
class ToFramesViewStageValidator(StageValidator):
    """validator for the to_frames view stage"""
    def __init__(self):
        super().__init__()
        self.stage_name = 'to_frames'
        self.num_positional_args = 1
        self.named_args = [
            "sample_frames", 
            "fps", 
            "max_fps", 
            "size", 
            "min_size", 
            "max_size", 
            "sparse", 
            "output_dir", 
            "rel_dir", 
            "frames_patt", 
            "force_samples", 
            "skip_failures", 
            "verbose"
            ]

    def validate_specific_stage(self, stage_args):
        """validate the to_frames view stage"""
        return True

###########################################################################




print('10 '.isdigit())

stage_str = 'limit(10)'
stage_str = 'filter_labels("frames.detections", F("label") == "vehicle")'
a, b, c = parse_view_stage(stage_str)
print(a, b, c)

# validate_view_stage(stage_str)

x = '"frames.detections", F("label") == "vehicle"'

# from fiftyone import ViewField as F
# a, b = x

# # vs = StageValidatorFactorySingleton()
# # print(vs)

# lvsv = LimitViewStageValidator()
# print(lvsv)

# l = StageValidatorFactory().get('limit')

# stage_inputs = {}
# print(l.validate(stage_inputs))
# # print(l)

# my_str = 'limit(10)'
# print(parse_view_stage(my_str))

# my_str = 'filter_labels("frames.detections", F("label") == "vehicle")'
# print(parse_view_stage(my_str))

# my_str = 'to_frames(sample_frames=True)'
# print(parse_view_stage(my_str))