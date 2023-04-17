from ast import parse
from ast2json import ast2json

# from fiftyone.core.stages import ViewStage
import fiftyone as fo

def convert_format(stage):
    words = stage.split('_')
    words = [word.capitalize() for word in words]
    return ''.join(words)

def validate_view_stage(stage_string, dataset):
    stage = stage_string.split('(')[0]
    stage = convert_format(stage)
    args = stage_string.split('(')[1].split(')')[0].split(',')

    try:
        eval(f"stage = fo.{stage}({args}).validate({dataset})")
    except:
        raise ValueError("SyntaxError: invalid Python syntax")
    # print(args)
    # print(stage)

#     try:
#         ast = parse(stage_string)
#     except SyntaxError:
#         raise ValueError("SyntaxError: invalid Python syntax")
    

#     ast_json = ast2json(ast)
#     body = ast_json["body"][0]["value"]
#     stage = body["func"]["id"]
#     args = body["args"]
#     keywords = body["keywords"]
#     stage_validator = StageValidatorFactory(dataset).get(stage)
#     stage_validator.validate(args, keywords)

# def validate_view_stages(stage_strings, dataset):
#     for stage_string in stage_strings:
#         validate_view_stage(stage_string, dataset)

# def validate_view_stage(stage_string, dataset):
#     ast_json = ast2json(ast)
#     body = ast_json["body"][0]["value"]
#     stage = body["func"]["id"]

    # try:
    #     ast = parse(stage_string)
    # except SyntaxError:
    #     raise ValueError("SyntaxError: invalid Python syntax")
    

# stage_string = 'exclude_fields("frames.detections", x = 3)'
# dataset = None
# validate_view_stage(stage_string, dataset)


















# from ast import parse
# from ast2json import ast2json

# KEYWORD_TYPES = {
#     "seed": "int",
#     "k": "int",
#     "tol": "int",
#     "min_len": "int",
#     "fps": "int",
#     "max_fps": "int",
#     "size": "int",
#     "min_size": "int",
#     "max_size": "int",
#     "bool": "bool",
#     "all": "bool",
#     "omit_empty": "bool",
#     "only_matches": "bool",
#     "ordered": "bool",
#     "reverse": "bool",
#     "keep_label_list": "bool",
#     "trajectories": "bool",
#     "sample_frames": "bool",
#     "sparse": "bool",
#     "force_samples": "bool",
#     "skip_failures": "bool",
#     "verbose": "bool",
#     "output_dir": "string",
#     "rel_dir": "string",
#     "frames_patt": "string",
#     "brain_key": "string",
#     "filter": "expression",
#     "config": "dict",
#     "location_field": "field",
#     "field_names": "field | field_list",
#     "dist_field": "field",
# }

# EXPRESSION_METHODS = {
#     "abs": {'args': [], 'kwargs': {}},
#     "all": {'args': ["expression_list"], 'kwargs': {}},
#     "any": {'args': ["expression_list"], 'kwargs': {}},
#     "append": {'args': ["const"], 'kwargs': {}},
#     "apply": {'args': ["expression"], 'kwargs': {}},
#     "arccos": {'args': [], 'kwargs': {}},
#     "arccosh": {'args': [], 'kwargs': {}},
#     "arcsin": {'args': [], 'kwargs': {}},
#     "arcsinh": {'args': [], 'kwargs': {}},
#     "arctan": {'args': [], 'kwargs': {}},
#     "arctanh": {'args': [], 'kwargs': {}},
#     "cases": {
#         'args': ["const_or_expression_dict"], 
#         'kwargs': {"default": "const_or_expression"}
#         },
#     "ceil": {'args': [], 'kwargs': {}},
#     "concat": {'args': ["string | expression_list"], 'kwargs': {"before": "bool"}},
#     "contains": {'args': ["string | string_list"], 'kwargs': {"all": "bool"}},
#     "contains_str": {
#         'args': ["string | string_list"], 
#         'kwargs': {"case_sensitive": "bool"}
#         },
#     "cos": {'args': [], 'kwargs': {}},
#     "cosh": {'args': [], 'kwargs': {}},
#     "day_of_month": {'args': [], 'kwargs': {}},
#     "day_of_week": {'args': [], 'kwargs': {}},
#     "day_of_year": {'args': [], 'kwargs': {}},
#     "difference": {'args': ["list"], 'kwargs': {}},
#     "ends_with": {
#         'args': ["string | string_list"], 
#         'kwargs': {"case_sensitive": "bool"}
#         },
#     "enumerate": {'args': ["list"], 'kwargs': {"start": "int"}},
#     "exists": {'args': [], 'kwargs': {"bool": "bool"}},
#     "exp": {'args': [], 'kwargs': {}},
#     "extend": {'args': ["expression | expression_list"], 'kwargs': {}},
#     "filter": {'args': ["expression"], 'kwargs': {}},
#     "floor": {'args': [], 'kwargs': {}},
#     "hour": {'args': [], 'kwargs': {}},
#     "if_else": {'args': ["expression", "expression"], 'kwargs': {}},
#     "insert": {'args': ["int", "const"], 'kwargs': {}},
#     "intersection": {'args': ["string_list | expression"], 'kwargs': {}},
#     "is_array": {'args': [], 'kwargs': {}},
#     "is_in": {'args': ["list"], 'kwargs': {}},
#     "is_missing": {'args': [], 'kwargs': {}},
#     "is_null": {'args': [], 'kwargs': {}},
#     "is_number": {'args': [], 'kwargs': {}},
#     "is_string": {'args': [], 'kwargs': {}},
#     "is_subset": {'args': ["list"], 'kwargs': {}},
#     "join": {'args': ["string"], 'kwargs': {}},
#     "length": {'args': [], 'kwargs': {}},
#     "let_in": {'args': ["expression"], 'kwargs': {}},
#     "literal": {'args': ["string"], 'kwargs': {}},
#     "ln": {'args': [], 'kwargs': {}},
#     "log": {'args': ["int"], 'kwargs': {}},
#     "log10": {'args': [], 'kwargs': {}},
#     "lower": {'args': [], 'kwargs': {}},
#     "lstrip": {'args': [], 'kwargs': {"chars": "string"}},
#     "map": {'args': ["expression"], 'kwargs': {}},
#     "map_values": {'args': ["dict"], 'kwargs': {}},
#     "matches_str": {
#         'args': ["string | string_list"], 
#         'kwargs': {"case_sensitive": "bool"}
#     },
#     "max": {'args': [], 'kwargs': {"value": "float"}},
#     "mean": {'args': [], 'kwargs': {}},
#     "millisecond": {'args': [], 'kwargs': {}},
#     "min": {'args': [], 'kwargs': {"value": "float"}},
#     "minute": {'args': [], 'kwargs': {}},
#     "month": {'args': [], 'kwargs': {}},
#     "pow": {'args': ["float"], 'kwargs': {}},
#     "prepend": {'args': ["const"], 'kwargs': {}},
#     "randn": {'args': [], 'kwargs': {}},
#     "range": {'args': ["int"], 'kwargs': {"stop": "int"}},
#     "re_match": {'args': ["string"], 'kwargs': {"options": "string"}},
#     "reduce": {'args': ["expression"], 'kwargs': {"init_val": "const"}},
#     "replace": {'args': ["string", "string"], 'kwargs': {}},
#     "reverse": {'args': [], 'kwargs': {}},
#     "round": {'args': [], 'kwargs': {"place": "int"}},
#     "rsplit": {'args': ["string"], 'kwargs': {"maxsplit": "int"}},
#     "rstrip": {'args': [], 'kwargs': {"chars": "string"}},
#     "second": {'args': [], 'kwargs': {}},
#     "set_equals": {'args': ["expression | expression_list"], 'kwargs': {}},
#     "set_field": {
#         'args': ["string", "const | expression"], 
#         'kwargs': {"relative": "bool"}
#         },
#     "sin": {'args': [], 'kwargs': {}},
#     "sinh": {'args': [], 'kwargs': {}},
#     "sort": {'args': [], 'kwargs': 
#              {"key":"const", "numeric":"bool", "reverse": "bool"}
#              },
#     "split": {'args': ["string"], 'kwargs': {"maxsplit": "int"}},
#     "sqrt": {'args': [], 'kwargs': {}},
#     "startswith": {
#         'args': ["string | string_list"], 
#         'kwargs': {"case_sensitive": "bool"}
#         },
#     "std": {'args': [], 'kwargs': {}},
#     "strip": {'args': [], 'kwargs': {"chars": "string"}},
#     "strlen": {'args': [], 'kwargs': {}},
#     "substr": {'args': ["int"], 'kwargs': {"start": "int", "end": "int", "count": "int"}},
#     "sum": {'args': [], 'kwargs': {}},
#     "switch":  {
#         'args': ["const | expression_dict"], 
#         'kwargs': {"default": "const | expression"}
#         },
#     "tan": {'args': [], 'kwargs': {}},
#     "tanh": {'args': [], 'kwargs': {}},
#     "to_bool": {'args': [], 'kwargs': {}},
#     "to_date": {'args': [], 'kwargs': {}},
#     "to_double": {'args': [], 'kwargs': {}},
#     "to_int": {'args': [], 'kwargs': {}},
#     "to_string": {'args': [], 'kwargs': {}},
#     "trunc": {'args': [], 'kwargs': {"place": "int"}},
#     "type": {'args': [], 'kwargs': {}},
#     "union": {'args': ["string_list_| expression"], 'kwargs': {}},
#     "unique": {'args': [], 'kwargs': {}},
#     "upper": {'args': [], 'kwargs': {}},
#     "week": {'args': [], 'kwargs': {}},
#     "year": {'args': [], 'kwargs': {}},
#     "zip": {
#         'args': ["string_list | expression"], 
#         'kwargs': {"use_longest": "bool", "defaults": "list"}
#         },
# }


# ###################################################################### 

# ######################################################################



# def validate_bool(aj, dataset):
#     if aj['_type'] == 'Constant' and type(aj['n']) == bool:
#         return True
#     else:
#         raise ValueError("Expected a boolean")

# def validate_string(aj, dataset):
#     if aj['_type'] == 'Constant' and type(aj['n']) == str:
#         return True
#     else:
#         raise ValueError("Expected a string")

# def validate_int(aj, dataset):
#     if aj['_type'] == 'Constant' and type(aj['n']) == int:
#         return True
#     else:
#         raise ValueError("Expected an integer")

# def validate_float(aj, dataset):
#     if aj['_type'] == 'Constant' and type(aj['n']) in [float, int]:
#         return True
#     else:
#         raise ValueError("Expected a float")

# def validate_const(aj, dataset):
#     if aj['_type'] == 'Constant':
#         return True
#     else:
#         raise ValueError("Expected a constant")

# def validate_dict(aj, dataset):
#     if aj['_type'] != 'Dict':
#         raise ValueError("Expected a dictionary")
    
#     for k in aj['keys']:
#         validate_const(k, dataset)
#     for v in aj['values']:
#         validate_const(v, dataset)

#     ### also validate that types of keys and values are the same
#     return True

# def validate_field(aj, dataset):
#     validate_string(aj, dataset)

#     field_name = aj['n']
#     if len(dataset.exists(field_name)) == 0:
#         raise ValueError("Field not found in dataset: {}".format(field_name))
#     return True

# def validate_string_dict(aj, dataset):
#     if aj['_type'] != 'Dict':
#         raise ValueError("Expected a dictionary")
#     for k in aj['keys']:
#         validate_string(k, dataset)
#     for v in aj['values']:
#         validate_string(v, dataset)
#     return True
    
# def validate_list(aj, dataset):
#     if aj['_type'] not in ['List', 'Tuple']:
#         raise ValueError("Expected a list")
    
#     for el in aj['elts']:
#         validate_const(el, dataset)
#     return True

# def validate_string_list(aj, dataset):
#     if aj['_type'] not in ['List', 'Tuple']:
#         raise ValueError("Expected a list")
    
#     for el in aj['elts']:
#         validate_string(el, dataset)
#     return True

# def validate_field_list(aj, dataset):
#     if aj['_type'] not in ['List', 'Tuple']:
#         raise ValueError("Expected a list")
    
#     for el in aj['elts']:
#         validate_field(el, dataset)
#     return True

# # def validate_string_or_string_list(aj, dataset):
# #     if aj['_type'] not in ['List', 'Tuple']:
# #         if aj['_type'] != 'Constant' or type(aj['n']) != str:
# #             raise ValueError("Expected a string or list of strings")
# #         else:
# #             return True
    
# #     for el in aj['elts']:
# #         validate_string(el, dataset)
# #     return True

# ######################################################################

# def validate_expression_base(aj, dataset):
#     if aj['func']['id'] == 'F':
#         if len(aj['args']) == 0:
#             return True
#         if len(aj['args']) == 1:
#             return validate_string(aj['args'][0], dataset)
#         else:
#             raise ValueError("ViewField F should have at most one argument")
#     else:
#         raise ValueError("ViewField should be a call to F")

# def validate_expression_args(args, expr_method, dataset):
#     if len(args) != len(EXPRESSION_METHODS[expr_method]['args']):
#         raise ValueError("ViewField method has incorrect number of arguments")
    
#     for i, arg in enumerate(args):
#         expected_types = EXPRESSION_METHODS[expr_method]['args'][i]
#         expected_types = [et.strip() for et in expected_types.split('|')]
#         validate_types(arg, expected_types, dataset)

# def validate_expression_keywords(keywords, expr_method, dataset):
#     for kw in keywords:
#         if kw['arg'] not in EXPRESSION_METHODS[expr_method]['kwargs']:
#             raise ValueError("ViewField method has incorrect keyword argument")
#         expected_types = EXPRESSION_METHODS[expr_method]['kwargs'][kw['arg']]
#         expected_types = [et.strip() for et in expected_types.split('|')]
#         validate_types(kw['value'], expected_types, dataset)

# def validate_attribute(aj, dataset):
#     attr = aj['func']['attr']
#     if attr not in list(EXPRESSION_METHODS.keys()):
#         raise ValueError("ViewField method not recognized")
    
#     args = aj['args']
#     keywords = aj['keywords']
#     validate_expression_args(args, attr, dataset)
#     validate_expression_keywords(keywords, attr, dataset)
#     return validate_expression(aj['func']['value'], dataset)

# def validate_binop(aj, dataset):
#     ### Boolean algebra OR on types (expression = 1, float = 0)
#     if aj['_type'] == 'BinOp':
#         left_type = validate_binop(aj['left'], dataset)
#         right_type = validate_binop(aj['right'], dataset)
#         return left_type or right_type
#     elif aj['_type'] == 'Call':
#         return validate_call(aj, dataset)
#     elif aj['_type'] == 'Constant':
#         return False
#     elif aj['_type'] == 'Compare':
#         left_type = validate_binop(aj['left'], dataset)
#         comp_type = validate_binop(aj['comparators'][0], dataset)
#         return left_type or comp_type
#     else:
#         raise ValueError("Invalid binary operation type")

# def validate_call(aj, dataset):
#     if aj['func']['_type'] == 'Name':
#         return validate_expression_base(aj, dataset)
#     else: ### Attribute
#         return validate_attribute(aj, dataset)
    
# def validate_expression_subroutine(aj, dataset):
#     ### Base case
#     if aj['_type'] == 'Call':
#         return validate_call(aj, dataset)
#     elif aj['_type'] == 'UnaryOp':
#         return validate_expression_subroutine(aj['operand'], dataset)
#     else: ### BinOp
#         return validate_binop(aj, dataset)
    
# def validate_expression(aj, dataset):
#     ### Recursion. Raise errors along the way if invalid. At end, if not an expression, raise error.
#     ve = validate_expression_subroutine(aj, dataset)
#     if not ve:
#         raise ValueError("Expected an expression")
#     return True
    
# def validate_expression_or_expression_list(aj, dataset):
#     if aj['_type'] not in ['List', 'Tuple']:
#         if aj['_type'] != 'Call':
#             raise ValueError("Expected an expression or list of expressions")
#         else:
#             return validate_expression(aj, dataset)
    
#     for el in aj['elts']:
#         validate_expression(el, dataset)
#     return True

# def validate_const_or_expression(aj, dataset):
#     return validate_const(aj, dataset) or validate_expression(aj, dataset)

# def validate_const_or_expression_dict(aj, dataset):
#     if aj['_type'] != 'Dict':
#         raise ValueError("Expected a dictionary")
    
#     for k in aj['keys']:
#         validate_const_or_expression(k, dataset)
#     for v in aj['values']:
#         validate_const_or_expression(v, dataset)

#     return True

# def validate_string_or_expression(aj, dataset):
#     if aj['_type'] == 'Constant' and type(aj['n']) == str:
#         return True
#     else:
#         return validate_expression(aj, dataset)

# def validate_string_or_expression_list(aj, dataset):
#     if aj['_type'] not in ['List', 'Tuple']:
#         return validate_string_or_expression(aj, dataset)
    
#     for el in aj['elts']:
#         validate_string_or_expression(el, dataset)
#     return True

# def validate_string_list_or_expression(aj, dataset):
#     if aj['_type'] not in ['List', 'Tuple']:
#         return validate_expression(aj, dataset)
#     else:
#         return validate_string_list(aj, dataset)

# ###############################################################################


    
# def validate_type(aj, expected_type, dataset):

#     router = {
#         'bool': validate_bool,
#         'string': validate_string,
#         'field': validate_field,
#         'int': validate_int,
#         'float': validate_float,
#         'const': validate_const,
#         'list': validate_list,
#         'string_list': validate_string_list,
#         'field_list': validate_field_list,
#         # 'string_or_string_list': validate_string_or_string_list,
#         'expression': validate_expression,
#         # "string_or_expression": validate_string_or_expression,
#         "string_dict": validate_string_dict,
#         "dict": validate_dict,
#         # "expression_or_expression_list": validate_expression_or_expression_list,
#         # "const_or_expression": validate_const_or_expression,
#         "const_or_expression_dict": validate_const_or_expression_dict,
#         "string_or_expression_list": validate_string_or_expression_list,
#         "string_list_or_expression": validate_string_list_or_expression
#     }

#     return router[expected_type](aj, dataset)

# def validate_types(aj, expected_types, dataset):
#     for et in expected_types:
#         if validate_type(aj, et, dataset):
#             return True
#     raise ValueError("Invalid type")

# ###############################################################################

# class StageValidatorFactory():
#     """factory for creating stage validators"""
#     def __init__(self, dataset):
#         self._validators = {
#             "exclude": ExcludeViewStageValidator(dataset),
#             "exclude_by": ExcludeByViewStageValidator(dataset),
#             "exclude_fields": ExcludeFieldsViewStageValidator(dataset),
#             "exclude_frames": ExcludeFramesViewStageValidator(dataset),
#             "exclude_labels": ExcludeLabelsViewStageValidator(dataset),
#             "exists": ExistsViewStageValidator(dataset),
#             "filter_field": FilterFieldViewStageValidator(dataset),
#             "filter_labels": FilterLabelsViewStageValidator(dataset),
#             "filter_keypoints": FilterKeypointsViewStageValidator(dataset),
#             "geo_near": GeoNearViewStageValidator(dataset),
#             "geo_within": GeoWithinViewStageValidator(dataset),
#             "limit": LimitViewStageValidator(dataset),
#             "limit_labels": LimitLabelsViewStageValidator(dataset),
#             "map_labels": MapLabelsViewStageValidator(dataset),
#             "set_field": SetFieldViewStageValidator(dataset),
#             "match": MatchViewStageValidator(dataset),
#             "match_frames": MatchFramesViewStageValidator(dataset),
#             "match_labels": MatchLabelsViewStageValidator(dataset),
#             "match_tags": MatchTagsViewStageValidator(dataset),
#             "select": SelectViewStageValidator(dataset),
#             "select_by": SelectByViewStageValidator(dataset),
#             "select_fields": SelectFieldsViewStageValidator(dataset),
#             "select_frames": SelectFramesViewStageValidator(dataset),
#             "select_labels": SelectLabelsViewStageValidator(dataset),
#             "shuffle": ShuffleViewStageValidator(dataset),
#             "skip": SkipViewStageValidator(dataset),
#             "sort_by": SortByViewStageValidator(dataset),
#             "sort_by_similarity": SortBySimilarityViewStageValidator(dataset),
#             "take": TakeViewStageValidator(dataset),
#             "to_patches": ToPatchesViewStageValidator(dataset),
#             "to_evaluation_patches": ToEvaluationPatchesViewStageValidator(dataset),
#             "to_clips": ToClipsViewStageValidator(dataset),
#             "to_trajectories": ToTrajectoriesViewStageValidator(dataset),
#             "to_frames": ToFramesViewStageValidator(dataset),
#         }

#     def get(self, name):
#         """get a stage validator"""
#         return self._validators[name]
    
# class StageValidator():
#     """base class for validating a view stage"""
#     def __init__(self, dataset):
#         self.dataset = dataset
#         pass
    
#     def validate_num_positional_args(self, positional_args):
#         return len(positional_args) == self.num_positional_args
    
#     def validate_stage_args(self, args):
#         if len(args) != len(self.arg_types):
#             raise ValueError(f"ViewStage {self.stage_name} has incorrect number of arguments")
        
#         for i, arg in enumerate(args):
#             expected_types = self.arg_types[i]
#             expected_types = [et.strip() for et in expected_types.split('|')]
#             validate_types(arg, expected_types, self.dataset)

#     def validate_stage_keywords(self, keywords):
#         for kw in keywords:
#             if kw['arg'] not in self.keywords:
#                 raise ValueError(f"ViewStage {self.stage_name} has incorrect keyword argument {kw['arg']}")
#             expected_types = KEYWORD_TYPES[kw['arg']]
#             expected_types = [et.strip() for et in expected_types.split('|')]
#             validate_types(kw['value'], expected_types, self.dataset)

#     def validate(self, args, keywords):
#         """validate a stage"""
#         self.validate_stage_args(args)
#         self.validate_stage_keywords(keywords)
#         self.validate_specific_stage(args, keywords)
    
#     def validate_specific_stage(self, args, keywords):
#         return "Not implemented"
    


# ###########################################################################

# class ExcludeViewStageValidator(StageValidator):
#     """validator for the exclude view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'exclude'
#         self.arg_types = ["string | string_list"]
#         self.keywords = []

#     def validate_specific_stage(self, args, keywords):
#         """validate the exclude view stage"""
#         return True

# class ExcludeByViewStageValidator(StageValidator):
#     """validator for the exclude_by view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'exclude_by'
#         self.arg_types = ["field", "list"]
#         self.keywords = []

#     def validate_specific_stage(self, args, keywords):
#         """validate the exclude_by view stage"""
#         return True

# class ExcludeFieldsViewStageValidator(StageValidator):
#     """validator for the exclude_fields view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'exclude_fields'
#         self.arg_types = ["field | field_list"]
#         self.keywords = []

#     def validate_specific_stage(self, args, keywords):
#         """validate the exclude_fields view stage"""
#         return True
    
# class ExcludeFramesViewStageValidator(StageValidator):
#     """validator for the exclude_frames view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'exclude_frames'
#         self.arg_types = ["string | string_list"]
#         self.keywords = ["omit_empty"]

#     def validate_specific_stage(self, args, keywords):
#         """validate the exclude_frames view stage"""
#         return True

# class ExcludeLabelsViewStageValidator(StageValidator):
#     """validator for the exclude_labels view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'exclude_labels'
#         self.arg_types = []
#         self.keywords = ["labels", "ids", "tags", "fields", "omit_empty"]

#     def validate_specific_stage(self, args, keywords):
#         """validate the exclude_labels view stage"""
#         return True    
    
# class ExistsViewStageValidator(StageValidator):
#     """validator for the exists view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'exists'
#         self.arg_types = ["string"]
#         self.keywords = ["bool"]

#     def validate_specific_stage(self, args, keywords):
#         """validate the exists view stage"""
#         return True
    
# class FilterFieldViewStageValidator(StageValidator):
#     """validator for the filter_field view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'filter_field'
#         self.arg_types = ["field", "expression"]
#         self.keywords = ["only_matches"]

#     def validate_specific_stage(self, args, keywords):
#         """validate the filter_field view stage"""
#         return True

# class FilterLabelsViewStageValidator(StageValidator):
#     """validator for the filter_labels view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'filter_labels'
#         self.arg_types = ["string", "expression"]
#         self.keywords = ["only_matches", "trajectories"]

#     def validate_specific_stage(self, args, keywords):
#         """validate the filter_labels view stage"""
#         return True
    
# class FilterKeypointsViewStageValidator(StageValidator):
#     """validator for the filter_keypoints view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'filter_keypoints'
#         self.arg_types = ["string"]
#         self.keywords = ["filter", "labels", "only_matches"]

#     def validate_specific_stage(self, args, keywords):
#         """validate the filter_keypoints view stage"""
#         return True
    
# class GeoNearViewStageValidator(StageValidator):
#     """validator for the geo_near view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'geo_near'
#         self.arg_types = ["list"]
#         self.keywords = ["location_field", "min_distance", "max_distance"]

#     def validate_specific_stage(self, args, keywords):
#         """validate the geo_near view stage"""
#         return True
    
# class GeoWithinViewStageValidator(StageValidator):
#     """validator for the geo_within view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'geo_within'
#         self.arg_types = ["list"]
#         self.keywords = ["location_field", "strict"]

#     def validate_specific_stage(self, args, keywords):
#         """validate the geo_within view stage"""
#         return True
    
# class LimitViewStageValidator(StageValidator):
#     """validator for the limit view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'limit'
#         self.arg_types = ["int"]
#         self.keywords = []

#     def validate_specific_stage(self, args, keywords):
#         """validate the limit_labels view stage"""
#         return True
    
# class LimitLabelsViewStageValidator(StageValidator):
#     """validator for the limit_labels view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'limit_labels'
#         self.arg_types = ["string", "int"]
#         self.keywords = []

#     def validate_specific_stage(self, args, keywords):
#         """validate the limit_labels view stage"""
#         return True
    
# class MapLabelsViewStageValidator(StageValidator):
#     """validator for the map_labels view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'map_labels'
#         self.arg_types = ["string", "string_dict"]
#         self.keywords = []

#     def validate_specific_stage(self, args, keywords):
#         """validate the map_labels view stage"""
#         return True

# class SetFieldViewStageValidator(StageValidator):
#     """validator for the set_field view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'set_field'
#         self.arg_types = ["field", "expression"]
#         self.keywords = []

#     def validate_specific_stage(self, args, keywords):
#         """validate the set_field view stage"""
#         return True
    
# class MatchViewStageValidator(StageValidator):
#     """validator for the match view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'match'
#         self.arg_types = ["expression"]
#         self.keywords = []

#     def validate_specific_stage(self, args, keywords):
#         """validate the match view stage"""
#         return True
    
# class MatchFramesViewStageValidator(StageValidator):
#     """validator for the match_frames view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'match_frames'
#         self.arg_types = ["expression"]
#         self.keywords = ["omit_empty"]

#     def validate_specific_stage(self, args, keywords):
#         """validate the match_frames view stage"""
#         return True
    
# class MatchLabelsViewStageValidator(StageValidator):
#     """validator for the match_labels view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'match_labels'
#         self.arg_types = []
#         self.keywords = ["labels", "ids", "tags", "filter", "fields", "bool"]

#     def validate_specific_stage(self, args, keywords):
#         """validate the match_labels view stage"""
#         return True

# class MatchTagsViewStageValidator(StageValidator):
#     """validator for the match_tags view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'match_tags'
#         self.arg_types = ["string | string_list"]
#         self.keywords = ["bool", "all"]

#     def validate_specific_stage(self, args, keywords):
#         """validate the match_tags view stage"""
#         return True
    
# class SelectViewStageValidator(StageValidator):
#     """validator for the select view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'select'
#         self.arg_types = ["string | string_list"]
#         self.keywords = ["ordered"]

#     def validate_specific_stage(self, args, keywords):
#         """validate the select view stage"""
#         return True

# class SelectByViewStageValidator(StageValidator):
#     """validator for the select_by view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'select_by'
#         self.arg_types = ["field", "list"]
#         self.keywords = ["ordered"]

#     def validate_specific_stage(self, args, keywords):
#         """validate the select_by view stage"""
#         return True

# class SelectFieldsViewStageValidator(StageValidator):
#     """validator for the select_fields view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'select_fields'
#         self.arg_types = []
#         self.keywords = ["field_names"]

#     def validate_specific_stage(self, args, keywords):
#         """validate the select_fields view stage"""
#         return True
    
# class SelectFramesViewStageValidator(StageValidator):
#     """validator for the select_frames view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'select_frames'
#         self.arg_types = ["string | string_list"]
#         self.keywords = ["omit_empty"]

#     def validate_specific_stage(self, args, keywords):
#         """validate the select_frames view stage"""
#         return True
    
# class SelectLabelsViewStageValidator(StageValidator):
#     """validator for the select_labels view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'select_labels'
#         self.arg_types = []
#         self.keywords = ["labels", "ids", "tags", "fields", "omit_empty"]

#     def validate_specific_stage(self, args, keywords):
#         """validate the select_labels view stage"""
#         return True

# class ShuffleViewStageValidator(StageValidator):
#     """validator for the shuffle view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'shuffle'
#         self.arg_types = []
#         self.keywords = ["seed"]

#     def validate_specific_stage(self, args, keywords):
#         """validate the shuffle view stage"""
#         return True
    
# class SkipViewStageValidator(StageValidator):
#     """validator for the skip view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'skip'
#         self.arg_types = []
#         self.keywords = []
    
# class SortByViewStageValidator(StageValidator):
#     """validator for the sort_by view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'sort_by'
#         self.arg_types = ["field | expression"]
#         self.keywords = ["reverse"]

#     def validate_specific_stage(self, args, keywords):
#         """validate the sort_by view stage"""
#         return True
    
# class SortBySimilarityViewStageValidator(StageValidator):
#     """validator for the sort_by_similarity view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'sort_by_similarity'
#         self.arg_types = ["string | string_list"]
#         self.keywords = ["k", "reverse", "dist_field", "brain_key"]

#     def validate_specific_stage(self, args, keywords):
#         """validate the sort_by_similarity view stage"""
#         return True
    
# class TakeViewStageValidator(StageValidator):
#     """validator for the take view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'take'
#         self.arg_types = ["int"]
#         self.keywords = ["seed"]

# class ToPatchesViewStageValidator(StageValidator):
#     """validator for the to_patches view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'to_patches'
#         self.arg_types = ["string"]
#         self.keywords = ["other_fields", "keep_label_list"]

#     def validate_specific_stage(self, args, keywords):
#         """validate the to_patches view stage"""
#         return True

# class ToClipsViewStageValidator(StageValidator):
#     """validator for the to_clips view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'to_clips'
#         self.arg_types = ["string | expression"]
#         self.keywords = ["other_fields", "tol", "min_len", "trajectories"]

#     def validate_specific_stage(self, args, keywords):
#         """validate the to_clips view stage"""
#         return True
    
# class ToEvaluationPatchesViewStageValidator(StageValidator):
#     """validator for the to_evaluation_patches view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'to_evaluation_patches'
#         self.arg_types = ["string"]
#         self.keywords = ["config"]

#     def validate_specific_stage(self, args, keywords):
#         """validate the to_evaluation_patches view stage"""
#         return True

# class ToTrajectoriesViewStageValidator(StageValidator):
#     """validator for the to_trajectories view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'to_trajectories'
#         self.arg_types = ["string"]
#         self.keywords = ["other_fields", "tol", "min_len", "trajectories"]

#     def validate_specific_stage(self, args, keywords):
#         """validate the to_trajectories view stage"""
#         return True
    
# class ToFramesViewStageValidator(StageValidator):
#     """validator for the to_frames view stage"""
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.stage_name = 'to_frames'
#         self.arg_types = []
#         self.keywords = [
#             "sample_frames", 
#             "fps", 
#             "max_fps", 
#             "size", 
#             "min_size", 
#             "max_size", 
#             "sparse", 
#             "output_dir", 
#             "rel_dir", 
#             "frames_patt", 
#             "force_samples", 
#             "skip_failures", 
#             "verbose"
#             ]

#     def validate_specific_stage(self, args, keywords):
#         """validate the to_frames view stage"""
#         return True

# ###########################################################################

# def validate_view_stage(stage_string, dataset):
#     try:
#         ast = parse(stage_string)
#     except SyntaxError:
#         raise ValueError("SyntaxError: invalid Python syntax")
    

#     ast_json = ast2json(ast)
#     body = ast_json["body"][0]["value"]
#     stage = body["func"]["id"]
#     args = body["args"]
#     keywords = body["keywords"]
#     stage_validator = StageValidatorFactory(dataset).get(stage)
#     stage_validator.validate(args, keywords)

# def validate_view_stages(stage_strings, dataset):
#     for stage_string in stage_strings:
#         validate_view_stage(stage_string, dataset)

# ######################################################################

# ### not handling Subscript - F(...)[0] yet
# ### not handling the "|" types yet
# ### not handling fields and label fields yet
# ##### Currently the EITHER | OR isn't working - it throws an error if any one of the types is wrong
# ## Doesn't handle label_type fields, 
# ## - exclude_labels, filter_labels, map_labels

# # stage_strings = ['']
# # stage_strings = ['filter_labels("frames.detections", F("label") >= (F("vehicle")>3))']
# # stage_strings = ['limit(5)']
# # stage_strings = ['limit(5)', 'shuffle(seed ="51", limit = 5)']
# # stage_strings = ['limit(5)', 'shuffle(seed =51, limit = 5)']
# # stage_strings = ['limit(5)', 'shuffle(seed =51)']
# # stage_strings = ['limit(F("vehicle")>3)', '("predictions", F("confidence") > 0.8)']
# # stage_strings = ['limit(10)', 'filter_labels("predictions", F("confidence") > 0.8)']
# # stage_strings = ['match(F("predictions.detections").filter(F() < 0.2).length() > 0)']


# # stage_strings = ['match(F("predictions.detections").ends_with("my_arg", case_sensitive = True) > 0)']

# # stage_strings = ['match(~F("predictions.detections").ends_with("my_arg", case_sensitive = True))']
# # stage_strings = ['match(F("predictions.detections").filter(F("a") >3).length() > 0)'] ### this is a bug in my expression validation
# # stage_strings = ['filter_labels("frames.detections", F("label") >= (F("vehicle")>3))']

# # stage_strings = ['filter_labels("predictions", F("bounding_box")[2] * F("bounding_box")[3] * F("$metadata.width")* F("$metadata.height") > 96**2)']
# # stage_strings = ['exclude_fields("frames.detections")']

# # validate_view_stages(stage_strings, dataset)



