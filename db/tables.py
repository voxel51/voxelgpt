
import uuid
import os
class UserQueryTable:
    def __init__(self, project_id, client, bigquery, dataset_id):
        table_id = os.environ.get('USER_QUERY_TABLE_ID', 'user_queries')
        self.table_id = f"{project_id}.{dataset_id}.{table_id}"
        self.client = client
        # Define the schema of the fields you want to insert
        self.schema = [
            bigquery.SchemaField("query_id", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("user_query", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("upvotes", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("downvotes", "INTEGER", mode="NULLABLE"),
        ]

        # Create the table if it doesn't exist
        try:
            self.client.get_table(self.table_id)
        except Exception as e:
            table = bigquery.Table(self.table_id, self.schema)
            self.client.create_table(table)

    def insert_query(self, user_query):
        # Generate a unique query_id using uuid
        query_id = str(uuid.uuid4())

        rows_to_insert = [(query_id, user_query, 0, 0)]

        # Insert rows into the table
        errors = self.client.insert_rows(self.client.get_table(self.table_id), rows_to_insert)

        if errors == []:
            print("New rows have been added.")
        else:
            msg = "Encountered errors while inserting rows: {}".format(errors)
            raise InsertExpection(msg)


    def upvote_query(self, query_id):
        # Update the upvotes count for the specified query
        query = f"""
        UPDATE `{self.table_id}`
        SET upvotes = upvotes + 1
        WHERE query_id = {query_id}
        """
        self.client.query(query)

    def downvote_query(self, query_id):
        # Update the downvotes count for the specified query
        query = f"""
        UPDATE `{self.table_id}`
        SET downvotes = downvotes + 1
        WHERE query_id = {query_id}
        """
        self.client.query(query)

class InsertExpection(Exception):
    pass

# Usage
# table = UserQueryTable('your_project_id', 'sample_dataset', 'queries_table')
# qid = table.insert_query("What's the weather like today?")
# table.upvote_query(qid)
# table.downvote_query(qid)


