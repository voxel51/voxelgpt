import os
import db.memory as memory

def get_client(project_id):
    if project_id:
       from google.cloud import bigquery
       return bigquery.Client(project=project_id)
    return memory.Client()

def get_ns(project_id):
    if project_id:
       from google.cloud import bigquery
       return bigquery
    return memory
