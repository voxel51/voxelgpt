import os

from .client import get_client, get_table
from .tables import UserQueryTable

def table(cls):
    return cls(get_client(), get_table(), os.environ.get('DATASET_ID'))