import os

from .client import get_client, get_ns
from .tables import UserQueryTable

def table(cls):
    project_id = os.environ.get('PROJECT_ID')
    return cls(project_id, get_client(project_id), get_ns(project_id), os.environ.get('DATASET_ID'))