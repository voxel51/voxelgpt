"""
Query DB tables.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import uuid
import os


class UserQueryTable(object):
    def __init__(self, project_id, client, bigquery, dataset_id):
        table_id = os.environ.get("USER_QUERY_TABLE_ID", "user_queries")
        self.table_id = f"{project_id}.{dataset_id}.{table_id}"
        self.client = client
        self.schema = [
            bigquery.SchemaField("query_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("user_query", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("upvotes", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("downvotes", "INTEGER", mode="NULLABLE"),
        ]

        # Create the table if it doesn't exist
        try:
            self.client.get_table(self.table_id)
        except Exception as e:
            table = bigquery.Table(self.table_id, self.schema)
            self.client.create_table(table)

    def _insert_row(self, query_id, user_query=None, upvotes=0, downvotes=0):
        errors = self.client.insert_rows(
            self.client.get_table(self.table_id),
            [(query_id, user_query, upvotes, downvotes)],
        )
        if errors:
            raise InsertExpection(
                f"Encountered errors while inserting rows: {errors}"
            )

    def insert_query(self, user_query):
        query_id = str(uuid.uuid4())
        self._insert_row(query_id, user_query=user_query)
        return query_id

    def upvote_query(self, query_id):
        self._insert_row(query_id, upvotes=1)

    def downvote_query(self, query_id):
        self._insert_row(query_id, downvotes=1)


class InsertExpection(Exception):
    pass
