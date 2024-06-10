"""
View creation classifier.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os
import requests
from typing import (
    List,
    Dict,
    Union,
    Optional,
    Literal,
)

from langchain_core.pydantic_v1 import BaseModel, Field

import fiftyone as fo
from fiftyone import ViewField as F

# pylint: disable=relative-beyond-top-level
from .utils import (
    PROMPTS_DIR,
    _build_chat_chain,
    gpt_4o,
    _make_replacements,
    _format_filter_expression,
)

stages_type = Optional[List[str]]
Number = Union[int, float]

## For debugging purposes
def write_log(log):
    with open("/tmp/log.txt", "a") as f:
        f.write(str(log) + "\n")


VIEW_STAGE_PROMPT_PREFIX = """
You are a helpful assistant for computer vision researchers and engineers using
the FiftyOne library. Your task is to help users create views in the FiftyOne App
by providing them with the appropriate `ViewStages` that can be used to filter,
sort, slice, match, and transform their datasets. For this task, you need to
help users """

LIMIT_STAGE_PROMPT_SUFFIX = """
Limit the number of samples in the view to the specified number. The `limit`
parameter specifies the maximum number of samples to include in the view.
"""

TAKE_STAGE_PROMPT_SUFFIX = """
Pick `take` random samples from the view. The `take` parameter specifies the number
of samples to take.
"""

SKIP_STAGE_PROMPT_SUFFIX = """
Skip the specified number of samples in the view. The `skip` parameter specifies
the number of samples to skip.
"""

SHUFFLE_STAGE_PROMPT_SUFFIX = """
Shuffle the samples in the view. This stage shuffles the samples in the view
randomly.
"""

EXISTS_STAGE_PROMPT_SUFFIX = """
Filters the samples in the current view to only include samples that have (or do not have)
a non-`None` value for the given field or embedded field.
"""

LIMIT_LABELS_STAGE_PROMPT_SUFFIX = """
Limits the number of `Label` instances in the specified labels list field of
each sample in the collection.
"""

SELECT_FIELDS_STAGE_PROMPT_SUFFIX = """
Select the specified fields in the current view. Only the selected fields (and
default fields like `id`, `tags`) will be present in the view.
"""

EXCLUDE_FIELDS_STAGE_PROMPT_SUFFIX = """
Excludes the specified fields in the current view. All fields except the excluded
fields will be present in the view.
"""

GEO_NEAR_STAGE_PROMPT_SUFFIX = """
Sort the samples in the view by their proximity to a specified geolocation,
optionally filtering by minimum and maximum distances.
"""

GEO_WITHIN_STAGE_PROMPT_SUFFIX = """
Filter the samples in the view to only include samples that are within a specified
geographical region.
"""

TO_PATCHES_STAGE_PROMPT_SUFFIX = """
Create a view that contains one sample per object patch in the
specified field of the collection.
"""

TO_EVALUATION_PATCHES_STAGE_PROMPT_SUFFIX = """
Creates a view based on the results of the evaluation with the
given key that contains one sample for each true positive, false
positive, and false negative example in the collection, respectively.

True positive examples will result in samples with both their ground
truth and predicted fields populated, while false positive/negative
examples will only have one of their corresponding predicted/ground
truth fields populated, respectively.

If multiple predictions are matched to a ground truth object (e.g., if
the evaluation protocol includes a crowd attribute), then all matched
predictions will be stored in the single sample along with the ground
truth object.

The returned view will also have top-level ``type`` and ``iou``
fields populated based on the evaluation results for that example, as
well as a ``sample_id`` field recording the sample ID of the example,
and a ``crowd`` field if the evaluation protocol defines a crowd
attribute.
"""

SELECT_BY_VIEW_STAGE_PROMPT_SUFFIX = """
Selects the samples with the given field values from the collection.

This stage is typically used to work with categorical fields (strings,
ints, and bools). If you want to select samples based on floating point
fields, use :meth:`match`.
"""

SELECT_GROUP_SLICES_STAGE_PROMPT_SUFFIX = """
Selects the samples in a group collection from the given slice(s).

The returned view is a flattened non-grouped view containing only the
slice(s) of interest.
"""

MATCH_TAGS_STAGE_PROMPT_SUFFIX = """
Returns a view containing the samples in the collection that have
or don't have any/all of the given tag(s).
"""

SELECT_LABELS_STAGE_PROMPT_SUFFIX = """
Selects only the specified labels from the collection.

The returned view will omit samples, sample fields, and individual
labels that do not match the specified selection criteria.
"""

SORT_BY_SIMILARITY_STAGE_PROMPT_SUFFIX = """
Sorts the samples in the view by their similarity to the specified text query.
"""

GROUP_BY_STAGE_PROMPT_SUFFIX = """
Groups the samples in the view by the specified field, embedded field, or
expression.
"""

SORT_BY_STAGE_PROMPT_SUFFIX = """
Sorts the samples in the view by the specified field, embedded field, or
expression. The `reverse` parameter specifies whether to sort in ascending (False)
or descending (True) order.
"""

FILTER_FIELD_STAGE_PROMPT_SUFFIX = """
Filters the samples in the view by the specified field and filtering expression.
Can be applied to fields of type `int`, `float`, `bool`, `str`, `date`, and
`datetime`.
"""

MATCH_LABELS_STAGE_PROMPT_SUFFIX = """
Filters the samples in the view to only include samples that have at least one
label that matches the described expression. This can be applied to all label
fields, or to specific label fields.
"""

FILTER_LABELS_STAGE_PROMPT_SUFFIX = """
Filters the samples in the view to only include the labels within the specified
label field that match the described expression.
"""

MAP_LABELS_STAGE_PROMPT_SUFFIX = """
Maps the labels in the specified label field of each sample in the collection
using the provided mapping dictionary.
"""

MATCH_STAGE_PROMPT_SUFFIX = """
Filters the samples in the view to only include samples that match the described
expression.
"""


VIEW_STAGE_PROMPTS = {
    "Limit": LIMIT_STAGE_PROMPT_SUFFIX,
    "Take": TAKE_STAGE_PROMPT_SUFFIX,
    "Skip": SKIP_STAGE_PROMPT_SUFFIX,
    "Shuffle": SHUFFLE_STAGE_PROMPT_SUFFIX,
    "Exists": EXISTS_STAGE_PROMPT_SUFFIX,
    "LimitLabels": LIMIT_LABELS_STAGE_PROMPT_SUFFIX,
    "SelectFields": SELECT_FIELDS_STAGE_PROMPT_SUFFIX,
    "ExcludeFields": EXCLUDE_FIELDS_STAGE_PROMPT_SUFFIX,
    "GeoNear": GEO_NEAR_STAGE_PROMPT_SUFFIX,
    "GeoWithin": GEO_WITHIN_STAGE_PROMPT_SUFFIX,
    "ToPatches": TO_PATCHES_STAGE_PROMPT_SUFFIX,
    "ToEvaluationPatches": TO_EVALUATION_PATCHES_STAGE_PROMPT_SUFFIX,
    "SelectBy": SELECT_BY_VIEW_STAGE_PROMPT_SUFFIX,
    "SelectGroupSlices": SELECT_GROUP_SLICES_STAGE_PROMPT_SUFFIX,
    "MatchTags": MATCH_TAGS_STAGE_PROMPT_SUFFIX,
    "SelectLabels": SELECT_LABELS_STAGE_PROMPT_SUFFIX,
    "SortBySimilarity": SORT_BY_SIMILARITY_STAGE_PROMPT_SUFFIX,
    "GroupBy": GROUP_BY_STAGE_PROMPT_SUFFIX,
    "SortBy": SORT_BY_STAGE_PROMPT_SUFFIX,
    "FilterField": FILTER_FIELD_STAGE_PROMPT_SUFFIX,
    "MatchLabels": MATCH_LABELS_STAGE_PROMPT_SUFFIX,
    "FilterLabels": FILTER_LABELS_STAGE_PROMPT_SUFFIX,
    "MapLabels": MAP_LABELS_STAGE_PROMPT_SUFFIX,
    "Match": MATCH_STAGE_PROMPT_SUFFIX,
}


class ViewStage(BaseModel):
    """View stage to apply to the view"""

    def build(self):
        raise NotImplementedError("Subclasses must implement this method")


class Take(ViewStage):
    """View stage to take random samples from the view

    Args:
        take: Number of samples to take

    Examples::

    # Take 10 random samples
    Take(take=10)

    """

    take: int = Field(description="Number of samples to take")

    def build(self):
        return fo.Take(self.take)

    def __repr__(self):
        return f"take({self.take})"


class Limit(ViewStage):
    """View stage to limit the number of samples in the view

    Args:
        limit: Maximum number of samples to include in the view

    Examples::

    # Limit the view to 100 samples
    Limit(limit=100)

    """

    limit: int = Field(
        description="Maximum number of samples to include in the view"
    )

    def build(self):
        return fo.Limit(self.limit)

    def __repr__(self):
        return f"limit({self.limit})"


class Skip(ViewStage):
    """View stage to skip the specified number of samples in the view

    Args:
        skip: Number of samples to skip

    Examples::

    # Skip the first 10 samples
    Skip(skip=10)

    """

    skip: int = Field(description="Number of samples to skip")

    def build(self):
        return fo.Skip(self.skip)

    def __repr__(self):
        return f"skip({self.skip})"


class Shuffle(ViewStage):
    """View stage to shuffle the samples in the view

    Args:
        seed: Seed for the random number generator

    Examples::

    # Shuffle the samples in the view
    Shuffle()

    """

    seed: Optional[int] = Field(
        description="Seed for the random number generator"
    )

    def build(self):
        return fo.Shuffle(seed=self.seed)

    def __repr__(self):
        return f"shuffle(seed={self.seed})"


class Exists(ViewStage):
    """View stage to filter samples that have (or do not have) a non-`None`
    value for the given field or embedded field.

    Args:
        field: Field or embedded field to check for existence
        positive_match: Whether to include samples that have (True) or do not
            have (False) a non-`None` value for the field

    Examples::

    # Only include samples that have a value in their `predictions` field
    Exists(field="predictions", positive_match=True)

    # Only include samples that do not have a value in their `field.embedded_field` field
    Exists(field="field.embedded_field", positive_match=False)

    """

    field: str = Field(
        description="Field or embedded field to check for existence"
    )
    positive_match: bool = Field(
        description="Whether to include samples that have (True) or do not have (False) a non-`None` value for the field"
    )

    def build(self):
        return fo.Exists(self.field, bool=self.positive_match)

    def __repr__(self):
        return f"exists({self.field}, bool={self.positive_match})"


class LimitLabels(ViewStage):
    """View stage to limit the number of `Label` instances in the specified
    labels list field of each sample in the collection.

    Args:
        field: Field to limit the number of `Label` instances
        limit: Maximum number of `Label` instances to include in the field

    Examples::

    # Only include the first detection in the `predictions` field of each sample
    LimitLabels(field="predictions", limit=1)

    # Only include the first 5 keypoints in the `keypoints` field
    LimitLabels(field="keypoints", limit=5)

    """

    field: str = Field(
        description="Field to limit the number of `Label` instances"
    )
    limit: int = Field(
        description="Maximum number of `Label` instances to include in the field"
    )

    def build(self):
        return fo.LimitLabels(self.field, self.limit)

    def __repr__(self):
        return f"limit_labels({self.field}, {self.limit})"


class SelectFields(ViewStage):
    """View stage to select the specified fields in the current view. Only the
    selected fields (and default fields like `id`, `tags`) will be present in
    the view.

    Args:
        fields: List of fields to select

    Examples::

    # Select the `field1` and `field2` fields
    SelectFields(fields=["field1", "field2"])

    """

    fields: List[str] = Field(description="List of fields to select")

    def build(self):
        return fo.SelectFields(self.fields)

    def __repr__(self):
        return f"select_fields(field_names={self.fields})"


class ExcludeFields(ViewStage):
    """View stage to exclude the specified fields in the current view. All fields
    except the excluded fields will be present in the view.

    Args:
        fields: List of fields to exclude

    Examples::

    # Exclude the `yolov8` and `yolov9` fields
    ExcludeFields(fields=["yolov8", "yolov9"])

    """

    fields: List[str] = Field(description="List of fields to exclude")

    def build(self):
        return fo.ExcludeFields(self.fields)

    def __repr__(self):
        return f"exclude_fields(field_names={self.fields})"


def _geocode_point(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json", "addressdetails": 1, "limit": 1}
    headers = {"User-Agent": "YourAppName/1.0 (your-email@example.com)"}

    response = requests.get(url, params=params, headers=headers)
    data = response.json()

    if data:
        first_place = data[0]
        latitude = first_place.get("lat")
        longitude = first_place.get("lon")
        return float(latitude), float(longitude)
    else:
        return None, None


class GeoNear(ViewStage):
    """View stage to sort the samples in the view by their proximity to a
    specified geolocation.

    Args:
        location_name: A string representing the place to sort by proximity to

    Examples::

    # Sort the samples by their proximity to the North Pole
    GeoNear(location_name="North Pole")

    # Find samples within 100 meters of the Eiffel Tower
    GeoNear(location_name="Eiffel Tower, Paris, France", max_distance=100)

    # Find samples between 50 and 100 meters of the Statue of Liberty
    GeoNear(location_name="Statue of Liberty, NY, USA", min_distance=50, max_distance=100)

    """

    location_name: str = Field(
        description="A string representing the place to sort by proximity to"
    )

    min_distance: Optional[Number] = Field(
        description="Minimum distance in meters"
    )
    max_distance: Optional[Number] = Field(
        description="Maximum distance in meters"
    )

    def build(self):
        latitude, longitude = _geocode_point(self.location_name)
        write_log(f"Latitude: {latitude}, Longitude: {longitude}")
        if latitude is None or longitude is None:
            raise ValueError(
                f"Could not geocode location: {self.location_name}"
            )
        return fo.GeoNear(
            [longitude, latitude],
            min_distance=self.min_distance,
            max_distance=self.max_distance,
        )

    def __repr__(self):
        return f"geo_near({self.location_name}, min_distance={self.min_distance}, max_distance={self.max_distance})"


def _geocode_boundary(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json", "polygon_geojson": 1, "limit": 1}
    headers = {"User-Agent": "YourAppName/1.0 (your-email@example.com)"}

    response = requests.get(url, params=params, headers=headers)
    data = response.json()

    if data:
        first_place = data[0]
        geojson = first_place.get("geojson")
        return geojson["coordinates"]
    else:
        return None


class GeoWithin(ViewStage):
    """View stage to filter the samples in the view to only include samples that
    are within a specified geographical region.

    Args:
        location_name: A string representing the region, city, or neighborhood
            to restrict the samples to

    Examples::

    # Filter for samples within Hell's Kitchen, New York
    GeoWithin(location_name="Hell's Kitchen, New York")

    # Filter for samples within Paris, France
    GeoWithin(location_name="Paris, France", max_distance=500)

    """

    location_name: str = Field(
        description="A string representing the place to filter by"
    )

    def build(self):
        boundary = _geocode_boundary(self.location_name)
        if boundary is None:
            raise ValueError(
                f"Could not geocode location: {self.location_name}"
            )
        return fo.GeoWithin(boundary)

    def __repr__(self):
        return f"geo_within({self.location_name})"


class ToPatches(ViewStage):
    """View stage to create a view that contains one sample per object patch in
    the specified field of the collection.

    Args:
        field: The field to extract patches from

    Examples::

    # Create a view containing one sample per object patch in the `ground_truth` field
    ToPatches(field="ground_truth")

    # Display the patches in the `predictions` field
    ToPatches(field="predictions")

    # Extract YOLOv8 detections
    ToPatches(field="yolov8")

    """

    field: str = Field(description="The field to extract patches from")

    def build(self):
        return fo.ToPatches(self.field)

    def __repr__(self):
        return f"to_patches({self.field})"


class ToEvaluationPatches(ViewStage):
    """View stage to create a view based on the results of the evaluation with
    the given key that contains one sample for each true positive, false positive,
    and false negative example in the collection, respectively.

    True positive examples will result in samples with both their ground truth and
    predicted fields populated, while false positive/negative examples will only
    have one of their corresponding predicted/ground truth fields populated,
    respectively.

    Args:
        eval_key: The evaluation key to use for the evaluation results

    Examples::

    # Create a view based on the evaluation results with the key "eval"
    ToEvaluationPatches(eval_key="eval")

    # Display the evaluation patches for the key "yolo_eval"
    ToEvaluationPatches(eval_key="yolo_eval")

    """

    eval_key: str = Field(
        description="The evaluation key to use for the evaluation results"
    )

    def build(self):
        return fo.ToEvaluationPatches(self.eval_key)

    def __repr__(self):
        return f"to_evaluation_patches({self.eval_key})"


class SelectBy(ViewStage):
    """View stage to select the samples with the given field values from the collection.

    Args:
        field: The field to select by
        values: The values to select
        ordered: Whether to preserve the order of the values in the output view (default is False)

    Examples::

    # Create a view containing samples whose `int` field has any of the values 1, 32, 51, or 100, preserving the order
    SelectBy(field="int", values=[1, 32, 51, 100], ordered=True)

    # Select samples with the `weather` field equal to "rainy" or "sunny"
    SelectBy(field="weather", values=["rainy", "sunny"])

    """

    field: str = Field(description="The field to select by")
    values: List[str] = Field(description="The values to select")

    ordered: bool = Field(
        description="Whether to preserve the order of the values in the output view (default is False)",
        default=False,
    )

    def build(self):
        return fo.SelectBy(self.field, self.values, ordered=self.ordered)

    def __repr__(self):
        return (
            f"select_by({self.field}, {self.values}, ordered={self.ordered})"
        )


class SelectGroupSlices(ViewStage):
    """View stage to select the samples in a group collection from the given
        slice(s). Either `slices` or `media_type` must be provided, but not both.

    Args:
        slices (None): a group slice or iterable of group slices to select.
            If neither argument is provided, a flattened list of all samples is
            returned
        media_type (None): a media type whose slice(s) to select

    Examples::

    # Retrieve the samples from the "left" or "right" group slices
    SelectGroupSlices(slices=["left", "right"])

    # Retrieve all image samples
    SelectGroupSlices(media_type="image")

    """

    slices: Optional[List[str]] = Field(
        description="A group slice or iterable of group slices to select"
    )

    media_type: Optional[Literal["image", "video", "3d"]] = Field(
        description="A media type whose slice(s) to select"
    )

    def build(self):
        return fo.SelectGroupSlices(
            slices=self.slices, media_type=self.media_type
        )

    def __repr__(self):
        return f"select_group_slices(slices={self.slices}, media_type={self.media_type})"


class MatchTags(ViewStage):
    """Return a view containing the samples in the collection that
    have or don't have any/all of the given tag(s).

    Args:
        tags: the tag or iterable of tags to match
        positive_match (True): whether to match samples that have (True) or
            do not have (False) the given tags
        match_all (False): whether to match samples that have (or don't have)
            all (True) or any (False) of the given tags

    Examples::

    # Only include samples that have the "test" tag
    MatchTags(tags=["test"], positive_match=True)

    # Only include samples that do not have the "test" tag
    MatchTags(tags=["test"], positive_match=False)

    # Only include samples that have the "test" or "train" tags
    MatchTags(tags=["test", "train"], positive_match=True)

    # Only include samples that have the "test" and "train" tags
    MatchTags(tags=["test", "train"], positive_match=True, match_all=True)

    # Only include samples that do not have the "test" or "train" tags
    MatchTags(tags=["test", "train"], positive_match=False)

    """

    tags: List[str] = Field(description="The tag or iterable of tags to match")
    positive_match: Optional[bool] = Field(
        description="Whether to match samples that have (True) or do not have (False) the given tags",
        default=True,
    )
    match_all: Optional[bool] = Field(
        description="Whether to match samples that have (or don't have) all (True) or any (False) of the given tags",
        default=False,
    )

    def build(self):
        return fo.MatchTags(
            tags=self.tags, bool=self.positive_match, all=self.match_all
        )

    def __repr__(self):
        return f"match_tags(tags={self.tags}, bool={self.positive_match}, all={self.match_all})"


class SelectLabels(ViewStage):
    """Selects only the specified labels from the collection.

    Args:
        tags: a tag or list of tags of labels to select
        fields (None): the field or list of fields from which to select labels

    Examples::

    # Retrieve all labels with the "test" tag
    SelectLabels(tags=["test"])

    # Retrieve all labels with the "v1" or "v2" tag from the "ground_truth" field
    SelectLabels(tags=["v1", "v2"], fields=["ground_truth"])

    """

    tags: Union[str, List[str]] = Field(
        description="The tag or list of tags of labels to select"
    )
    fields: Optional[List[str]] = Field(
        description="The field or list of fields from which to select labels"
    )

    def build(self):
        return fo.SelectLabels(tags=self.tags, fields=self.fields)

    def __repr__(self):
        return f"select_labels(tags={self.tags}, fields={self.fields})"


class SortBySimilarity(ViewStage):
    """Sorts the samples in the view by their similarity to the specified text query.

    Args:
        text: The text query to sort by
        k (None): The number of samples to return. If None, all samples are returned
        brain_key (None): The key of the brain to use for similarity search

    Examples::

    # Return the 10 most similar samples to the text "animal"
    SortBySimilarity(text="animal")

    # Sort by similarity to the text "kites flying in the sky" with brain key "clip_sim"
    SortBySimilarity(text="kites flying in the sky", brain_key="clip_sim")

    """

    text: str = Field(description="The text query to sort by")
    k: Optional[int] = Field(
        description="The number of samples to return. If None, all samples are returned",
        default=25,
    )
    brain_key: Optional[str] = Field(
        description="The key of the brain to use for similarity search"
    )

    def build(self):
        return fo.SortBySimilarity(
            self.text, k=self.k, brain_key=self.brain_key
        )

    def __repr__(self):
        return f"sort_by_similarity({self.text}, k={self.k}, brain_key={self.brain_key})"


class GroupBy(ViewStage):
    """Groups the samples in the view by the specified field, embedded field, or
    expression.

    Args:
        key: The field, embedded field, or expression to group by. Must start with 'F('

    Examples::

    # Group the samples by their `weather` field
    GroupBy(key='F("weather")')

    # Group images by width
    GroupBy(key='F("IMAGE_WIDTH")')

    # Group samples by their class labels in the predictions field
    GroupBy(key='F("predictions.label")')

    # Group by the number of ground truth detections
    GroupBy(key='F("ground_truth.detections").length()')

    # Group by the number of false positives
    GroupBy(key='F("eval_fp")')

    """

    key: str = Field(
        description="The field, embedded field, or expression to group by"
    )

    def build(self):
        # pylint: disable=no-member
        if not self.key.startswith("F(") and not self.key.startswith('F("'):
            key = 'F("' + self.key + '")'
        key = _make_replacements(self.key)
        key = eval(key)
        if isinstance(key, str):
            key = eval(key)

        return fo.GroupBy(key)

    def __repr__(self):
        # pylint: disable=no-member
        if not self.key.startswith("F(") and not self.key.startswith('F("'):
            key = 'F("' + self.key + '")'
        key = _make_replacements(key)
        return f"group_by({key})"


class SortBy(ViewStage):
    """Sorts the samples in the view by the specified field, embedded field, or
    expression.

    Args:
        key: The field, embedded field, or expression to sort by. Must start with 'F('
        reverse (False): Whether to sort in descending order

    Examples::

    # Sort images by width
    SortBy(key='F("IMAGE_WIDTH")')

    # Sort samples by their uniqueness score, with most unique samples first
    SortBy(key='F("uniqueness")', reverse=True)

    # Sort by the number of ground truth detections
    SortBy(key='F("ground_truth.detections").length()')

    # Sort by the number of false positives
    SortBy(key='F("eval_fp")')

    """

    key: str = Field(
        description="The field, embedded field, or expression to sort by"
    )
    reverse: bool = Field(
        description="Whether to sort in descending order", default=False
    )

    def build(self):
        # pylint: disable=no-member
        if not self.key.startswith("F(") and not self.key.startswith('F("'):
            key = 'F("' + self.key + '")'
        key = _make_replacements(self.key)
        key = eval(key)
        if isinstance(key, str):
            key = eval(key)

        return fo.SortBy(key, reverse=self.reverse)

    def __repr__(self):
        # pylint: disable=no-member
        if not self.key.startswith("F(") and not self.key.startswith('F("'):
            key = 'F("' + self.key + '")'
        key = _make_replacements(key)
        return f"sort_by({key}, reverse={self.reverse})"


class FilterField(ViewStage):
    """Filters the values of a field of each sample in the collection.

    Args:
        field: The field to filter by
        filter_expression: The description of the filter

    Examples::

    # Only include samples whose "uniqueness" is positive
    FilterField(field="uniqueness", filter_expression="only include positive values")

    # Only include samples whose "date" field is greater than 2022
    FilterField(field="date", filter_expression="only include dates greater than 2022")

    # Filter for samples with either 3 or 4 FPs
    FilterField(field="eval_fp", filter_expression="only include samples with 3 or 4 FPs")
    """

    field: str = Field(description="The field to filter by")
    filter_expression: str = Field(description="The description of the filter")

    def build(self):
        filter_expr = _format_filter_expression(self.filter_expression)
        if self.field == "filepath":
            ## Edge case, switch to fo.Match()
            ## Insert filepath into F() expression
            filter_expr = filter_expr.replace("F()", 'F("filepath")')
            return fo.Match(eval(filter_expr))
        return fo.FilterField(self.field, eval(filter_expr))

    def __repr__(self):
        return f"filter_field({self.field}, {self.filter_expression})"


class MatchLabels(ViewStage):
    """Selects samples from the collection containing at least one label
    that satisfies the specified filter expression.

    If you use `positive=False`, you must invert the filter expression to match

    Args:
        filter_expression: The description of the filter to apply to the labels
        fields (None): The label field(s) on which to match. If None, all fields
            are considered
        positive (True): Whether to match samples that have (True) or do not have (False)
            a label satisfying the filter

    Examples::

    # Only include samples with a ground truth dog detection
    MatchLabels(
        fields="ground_truth",
        filter_expression="samples with a dog detection"
        )

    # Only include samples w/o a high confidence prediction
    MatchLabels(
        fields="predictions",
        filter_expression="samples with a high confidence prediction"
        positive=False
        )

    # Only include samples with bboxes that are >= 50% of the image size
    MatchLabels(
        filter_expression="bounding boxes at least 50% of the image size"
        )
    """

    fields: Optional[Union[str, List[str]]] = Field(
        description="The field or list of fields from which to select labels"
    )
    filter_expression: str = Field(
        description="The description of the filter to apply to the labels"
    )
    positive: bool = Field(
        description="Whether to match samples that have (True) or do not have (False) a label that satisfies the filter expression",
        default=True,
    )

    def build(self):
        filter_expr = _format_filter_expression(self.filter_expression)
        filter_expr = eval(filter_expr)
        return fo.MatchLabels(
            fields=self.fields, filter=filter_expr, bool=self.positive
        )

    def __repr__(self):
        return f"match_labels(fields={self.fields}, filter={self.filter_expression}, bool={self.positive})"


class FilterLabels(ViewStage):
    """Filters the contents of a label field for labels that match the specified
        filter expression.

    Args:
        field: The label field to filter
        filter_expression: The description of the filter to apply to the labels

    Examples::

    # Only include high confidence predictions
    FilterLabels(
        field="predictions",
        filter_expression="high confidence predictions"
        )

    # Only include GT detections with bboxes that are >= 50% of the image size
    FilterLabels(
        field="ground_truth",
        filter_expression="bounding boxes at least 50% of the image size"
        )
    """

    field: str = Field(description="The field from which to select labels")
    filter_expression: str = Field(
        description="The description of the filter to apply to the labels"
    )

    def build(self):
        filter_expr = _format_filter_expression(self.filter_expression)
        filter_expr = eval(filter_expr)
        return fo.FilterLabels(self.field, filter_expr)

    def __repr__(self):
        filter_expr = _format_filter_expression(self.filter_expression)
        return f"filter_labels({self.field}, {filter_expr})"


class MapLabels(ViewStage):
    """Maps the labels in the specified label field of each sample in the
    via a mapping dictionary.

    Args:
        field: The label field to map
        mapping: The dictionary mapping old values to new values

    Examples::

    # Map the "dog" label to "canine" in the "ground_truth" field
    MapLabels(
        field="ground_truth",
        mapping={"dog": "canine"}
        )

    # Map the "cat" label to "feline" and the "dog" label to "canine" in the "predictions" field
    MapLabels(
        field="predictions",
        mapping={"cat": "feline", "dog": "canine"}
        )
    """

    field: str = Field(description="The label field to map")
    mapping: Dict[str, str] = Field(
        description="The dictionary mapping old values to new values"
    )

    def build(self):
        return fo.MapLabels(self.field, self.mapping)

    def __repr__(self):
        return f"map_labels({self.field}, {self.mapping})"


class Match(ViewStage):
    """
    Filters the samples in the collection by the given filter.

    Args:
        filter_expression: The description of the filter to apply to the samples

    Examples::

    # Only include samples with at least 2 objects in their `predictions` field
    Match(filter_expression="samples with at least 2 objects in their predictions")

    # Filter for samples that only contain dogs in their `ground_truth` field
    Match(filter_expression="samples that only contain dogs in their ground truth")

    # Only include samples with a dog and a cat prediction but no bird prediction
    Match(filter_expression="samples with a dog and a cat prediction but no bird prediction")
    """

    filter_expression: str = Field(
        description="The description of the filter to apply to the samples"
    )

    def build(self):
        filter_expr = _format_filter_expression(self.filter_expression)
        filter_expr = eval(filter_expr)
        return fo.Match(filter_expr)

    def __repr__(self):
        filter_expr = _format_filter_expression(self.filter_expression)
        return f"match({filter_expr})"


VIEW_STAGE_OUTPUT_TYPES = {
    "Limit": Limit,
    "Take": Take,
    "Skip": Skip,
    "Shuffle": Shuffle,
    "Exists": Exists,
    "LimitLabels": LimitLabels,
    "SelectFields": SelectFields,
    "ExcludeFields": ExcludeFields,
    "GeoNear": GeoNear,
    "GeoWithin": GeoWithin,
    "ToPatches": ToPatches,
    "ToEvaluationPatches": ToEvaluationPatches,
    "SelectBy": SelectBy,
    "SelectGroupSlices": SelectGroupSlices,
    "MatchTags": MatchTags,
    "SelectLabels": SelectLabels,
    "SortBySimilarity": SortBySimilarity,
    "GroupBy": GroupBy,
    "SortBy": SortBy,
    "FilterField": FilterField,
    "MatchLabels": MatchLabels,
    "FilterLabels": FilterLabels,
    "MapLabels": MapLabels,
    "Match": Match,
}


### FILTER FIELD EXPRESSION CONSTRUCTION ###

FILTER_FIELD_EXPRESSION_PROMPT_FILENAMES = {
    "str": "filter_field_string_expression.txt",
    "int": "filter_field_int_expression.txt",
    "float": "filter_field_float_expression.txt",
    "bool": "filter_field_bool_expression.txt",
    "date": "filter_field_date_expression.txt",
    "datetime": "filter_field_datetime_expression.txt",
    "list": "filter_field_list_expression.txt",
    "other": "filter_field_other_expression.txt",
}


def _construct_filter_field_expression(
    stage, step, inspection_results, dataset
):
    field = dataset.get_field(stage.field)

    if isinstance(field, fo.StringField):
        prompt_filename = FILTER_FIELD_EXPRESSION_PROMPT_FILENAMES["str"]
    elif isinstance(field, fo.IntField):
        prompt_filename = FILTER_FIELD_EXPRESSION_PROMPT_FILENAMES["int"]
    elif isinstance(field, fo.FloatField):
        prompt_filename = FILTER_FIELD_EXPRESSION_PROMPT_FILENAMES["float"]
    elif isinstance(field, fo.BooleanField):
        prompt_filename = FILTER_FIELD_EXPRESSION_PROMPT_FILENAMES["bool"]
    elif isinstance(field, fo.DateField):
        prompt_filename = FILTER_FIELD_EXPRESSION_PROMPT_FILENAMES["date"]
    elif isinstance(field, fo.DateTimeField):
        prompt_filename = FILTER_FIELD_EXPRESSION_PROMPT_FILENAMES["datetime"]
    elif isinstance(field, fo.ListField):
        prompt_filename = FILTER_FIELD_EXPRESSION_PROMPT_FILENAMES["list"]
    else:
        prompt_filename = FILTER_FIELD_EXPRESSION_PROMPT_FILENAMES["other"]

    FILTER_FIELD_EXPRESSION_PATH = os.path.join(PROMPTS_DIR, prompt_filename)

    chain = _build_chat_chain(
        gpt_4o, template_path=FILTER_FIELD_EXPRESSION_PATH
    )

    resp = chain.invoke({"messages": [("user", step)]}).content
    stage.filter_expression = resp


MATCH_LABELS_EXPRESSION_PROMPT_FILENAMES = {
    "general": "match_labels_general_expression.txt",
    "classification": "match_labels_general_expression.txt",  ## if needed, make a separate one
    "detections_2d": "match_labels_detections_2d_expression.txt",
    "detections_3d": "match_labels_general_expression.txt",  #! TO DO: Make a separate one
}


def _identify_label_field_type(dataset, field_names, filter_expression):
    if (
        dataset.media_type == "group"
        and "3d" in dataset.group_media_types.values()
    ):
        if "volume" in filter_expression or "rotation" in filter_expression:
            return "detections_3d"
    num_det_fields = len(
        dataset.get_field_schema(embedded_doc_type=fo.Detections)
    )
    num_cls_fields = len(
        dataset.get_field_schema(embedded_doc_type=fo.Classification)
    )

    if num_cls_fields == 0:
        return "detections_2d"
    if num_det_fields == 0:
        return "classification"

    if field_names is not None and len(field_names) == 1:
        field = dataset.get_field(field_names[0])
        if isinstance(field, fo.Detections):
            return "detections_2d"
        elif isinstance(field, fo.Classification):
            return "classification"

    det2d_phrases = ["bounding box", "detection", "object", "area", "bbox"]
    if any(phrase in filter_expression for phrase in det2d_phrases):
        return "detections_2d"

    return "general"


def _construct_match_labels_expression(
    stage, step, inspection_results, dataset
):
    field = stage.field if hasattr(stage, "field") else stage.fields
    label_type = _identify_label_field_type(dataset, field, step)
    prompt_filename = MATCH_LABELS_EXPRESSION_PROMPT_FILENAMES[label_type]
    MATCH_LABELS_EXPRESSION_PATH = os.path.join(PROMPTS_DIR, prompt_filename)

    chain = _build_chat_chain(
        gpt_4o, template_path=MATCH_LABELS_EXPRESSION_PATH
    )

    resp = chain.invoke({"messages": [("user", step)]}).content
    stage.filter_expression = resp


def _construct_match_expression(stage, step, inspection_results, dataset):
    FILTER_FIELD_EXPRESSION_PATH = os.path.join(
        PROMPTS_DIR, "match_expression.txt"
    )

    chain = _build_chat_chain(
        gpt_4o, template_path=FILTER_FIELD_EXPRESSION_PATH
    )

    resp = chain.invoke({"messages": [("user", step)]}).content
    stage.filter_expression = resp


def _construct_view_expression_if_needed(
    stage, step, inspection_results, dataset
):
    if isinstance(stage, FilterField):
        _construct_filter_field_expression(
            stage, step, inspection_results, dataset
        )
    elif isinstance(stage, MatchLabels) or isinstance(stage, FilterLabels):
        ## Handle MatchLabels and FilterLabels together
        _construct_match_labels_expression(
            stage, step, inspection_results, dataset
        )
    elif isinstance(stage, Match):
        _construct_match_expression(stage, step, inspection_results, dataset)


def construct_stage(step, assignee, inspection_results, dataset):
    PROMPT_SUFFIX = VIEW_STAGE_PROMPTS[assignee]
    prompt = VIEW_STAGE_PROMPT_PREFIX + PROMPT_SUFFIX
    ### Add data inspection results...

    output_type = VIEW_STAGE_OUTPUT_TYPES[assignee]

    chain = _build_chat_chain(gpt_4o, prompt=prompt, output_type=output_type)
    stage = chain.invoke({"messages": [("user", step)]})
    _construct_view_expression_if_needed(
        stage, step, inspection_results, dataset
    )
    return stage
