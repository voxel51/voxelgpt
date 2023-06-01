from datetime import date, datetime
import fiftyone as fo
from random import randint, choice, random


def get_date_fields(dataset):
    fields = dataset.get_field_schema(flat=True)
    return [
        field_name
        for field_name, field in fields.items()
        if type(field) == fo.core.fields.DateField
    ]


def get_datetime_fields(dataset):
    fields = dataset.get_field_schema(flat=True)
    return [
        field_name
        for field_name, field in fields.items()
        if type(field) == fo.core.fields.DateTimeField
    ]


def get_string_fields(dataset):
    fields = dataset.get_field_schema(flat=True)
    return [
        field_name
        for field_name, field in fields.items()
        if type(field) == fo.core.fields.StringField
        and "." not in field_name
        and field_name != "filepath"
    ]


def get_label_fields(dataset):
    det_fields = []
    classification_fields = []
    classifications_fields = []
    sample = dataset.first()
    for field_name, field in sample.iter_fields():
        if type(field) == fo.core.labels.Detections:
            det_fields.append(field_name)
        elif type(field) == fo.core.labels.Classification:
            classification_fields.append(field_name)
        elif type(field) == fo.core.labels.Classifications:
            classifications_fields.append(field_name)
    return det_fields, classification_fields, classifications_fields


def get_dataset_field_types(dataset):
    string_fields = get_string_fields(dataset)
    date_fields = get_date_fields(dataset)
    datetime_fields = get_datetime_fields(dataset)
    (
        det_fields,
        classification_fields,
        classifications_fields,
    ) = get_label_fields(dataset)

    return {
        "string_fields": string_fields,
        "date_fields": date_fields,
        "datetime_fields": datetime_fields,
        "det_fields": det_fields,
        "classification_fields": classification_fields,
        "classifications_fields": classifications_fields,
    }


BASE_FIELD_PATTERNS = [
    {
        "query": "Exclude the {FIELD} field from all samples",
        "stages": "[exclude_fields('{FIELD}')]",
    },
    {
        "query": "Only show samples with field {FIELD}",
        "stages": "[exists('{FIELD}')]",
    },
    {
        "query": "Just show field {FIELD}",
        "stages": "[select_fields('{FIELD}')]",
    },
]

STRING_PATTERNS = [
    {
        "query": "exclude samples with {FIELD} in {VALUE_LIST}",
        "stages": "[exclude_by('{FIELD}', {VALUE_LIST})]",
    },
    {
        "query": "Images where {FIELD} is {VALUE}",
        "stages": "[match(F('{FIELD}') == '{VALUE}')]",
    },
    {
        "query": "Only images that have {FIELD} not equal to {VALUE}",
        "stages": "[match(F('{FIELD}') != '{VALUE}')]",
    },
    {
        "query": "Show me all the images where {FIELD} ends with s",
        "stages": "[match(F('{FIELD}').ends_with('s'))]",
    },
    {
        "query": " images where {FIELD} is {VALUE1} or {VALUE2}",
        "stages": "[match(F('{FIELD}').is_in(['{VALUE1}', '{VALUE2}']))]",
    },
]

DATE_PATTERNS = [
    {
        "query": "Show me all the images with {FIELD} before {VALUE}",
        "stages": "[match(F('{FIELD}') < {VALUE})]",
    },
    {
        "query": "Show me all the images with {FIELD} after {VALUE}",
        "stages": "[match(F('{FIELD}') > {VALUE})]",
    },
    {
        "query": "Samples where {FIELD} is in February",
        "stage": "[match(F('{FIELD}').month() == 2)]",
    },
    {
        "query": "Samples that have 1988 for {FIELD}",
        "stage": "[match(F('{FIELD}').year() == 1988)]",
    },
    {
        "query": "Any images with {FIELD} in the last 10 years",
        "stage": "[match(F('{FIELD}').year() > 2013)]",
    },
    {
        "query": "All of the samples with {FIELD} first five days of the month",
        "stage": "[match(F('{FIELD}').day_of_month() < 6)]",
    },
]

DATETIME_PATTERNS = DATE_PATTERNS + [
    {
        "query": "Images where the minute for {FIELD} equals 30",
        "stage": "[match(F('{FIELD}').minute() == 30)]",
    },
    {
        "query": "Images with field {FIELD} after 6pm",
        "stage": "[match(F('{FIELD}').hour() > 18)]",
    },
    {
        "query": "Display the samples where {FIELD} has millisecond of 3 or 4",
        "stage": "[match(F('{FIELD}').millisecond().is_in([3, 4]))]",
    },
]


class FieldExampleGenerator(object):
    """Base class for generating synthetic examples."""

    def __init__(self, dataset, field_name):
        self.dataset = dataset
        self.field_name = field_name
        self.filters = {
            "geo": False,
            "text_sim": False,
            "image_sim": False,
            "eval": False,
            "metadata": False,
            "label_types": ["all"],
        }

    def get_field_values(self):
        raise NotImplementedError()

    def generate_example(self):
        raise NotImplementedError()


class StringFieldExampleGenerator(FieldExampleGenerator):
    def __init__(self, dataset, field_name):
        super().__init__(dataset, field_name)
        self.field_values = self.get_field_values()

    def get_field_values(self):
        return self.dataset.distinct(self.field_name)

    def generate_example(self):
        return self.field_values[0]
