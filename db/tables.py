
import uuid
import os
class UserQueryTable:
    def __init__(self, project_id, client, bigquery, dataset_id):
        table_id = os.environ.get('USER_QUERY_TABLE_ID', 'user_queries')
        self.table_id = f"{project_id}.{dataset_id}.{table_id}"
        print(f"Table ID: {self.table_id}")
        self.client = client
        # Define the schema of the fields you want to insert
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
        full_row = (query_id, user_query, upvotes, downvotes)
        errors = self.client.insert_rows(self.client.get_table(self.table_id), [full_row])
        if errors == []:
            return
        else:
            print(errors)
            msg = "Encountered errors while inserting rows: {}".format(errors)
            raise InsertExpection(msg)

    def insert_query(self, user_query):
        # Generate a unique query_id using uuid
        query_id = str(uuid.uuid4())

        # Persist the user query to the table
        self._insert_row(query_id, user_query=user_query)

        return query_id

    def upvote_query(self, query_id):
        self._insert_row(query_id, upvotes=1)

    def downvote_query(self, query_id):
        self._insert_row(query_id, downvotes=1)

class InsertExpection(Exception):
    pass
