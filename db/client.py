import os

def get_client():
    project_id = os.environ.get('PROJECT_ID')
    if project_id:
       from google.cloud import bigquery
       return bigquery.Client(project=project_id)
    return MemoryClient()

def get_table():
    project_id = os.environ.get('PROJECT_ID')
    if project_id:
       from google.cloud import bigquery
       return bigquery.Table
    return MemoryTable

class MemoryClient:
    def __init__(self):
        self.tables = {}

    def get_table(self, table_id):
        return self.tables.get(table_id)

    def create_table(self, table):
        self.tables[table.table_id] = table

    def insert_rows(self, table, rows_to_insert):
        table.insert_rows(rows_to_insert)

    def query(self, query):
        parsed = parse_sql(query)
        table = self.get_table(parsed["table_id"])
        if parsed.get("set", None) and parsed.get("where", None):
            table.update_rows(parsed["where"], parsed["set"])
    
class MemoryTable:
    def __init__(self, table_id, schema):
        self.table_id = table_id
        self.schema = schema
        self.rows = []

    def insert_rows(self, rows_to_insert):
        self.rows.extend(rows_to_insert)

    def _get_idx_for_column(self, column):
        for idx, field in enumerate(self.schema):
            if field.name == column:
                return idx
        return None
    
    def _get_value_for_column(self, row, column):
        idx = self._get_idx_for_column(column)
        return row[idx]

    def update_rows(self, where, set):
        for row in self.rows:
            column = where["column"]
            value = where["value"]
            column_value = self._get_value_for_column(row, column)
            if column_value == value:
                if set["increment"]:
                    idx = self._get_idx_for_column(set["column"])
                    row[idx] = row[idx] + set["increment"]
                if set["decrement"]:
                    idx = self._get_idx_for_column(set["column"])
                    row[idx] = row[idx] - set["increment"]
                return

def parse_sql(sql):
    # Define regex patterns
    table_pattern = r"UPDATE `(.+?)`"
    set_pattern = r"SET (\w+) = (\w+) \+ (\d+)"
    where_pattern = r"WHERE (\w+) = (\w+)"
    
    # Extract values
    table_id = re.search(table_pattern, sql).group(1)
    set_clause = re.search(set_pattern, sql).groups()
    where_clause = re.search(where_pattern, sql).groups()
    
    # Return as dictionary
    return {
        "table_id": table_id,
        "set": {
            "column": set_clause[0],
            "base": set_clause[1],
            "increment": int(set_clause[2])
        },
        "where": {
            "column": where_clause[0],
            "value": where_clause[1]
        }
    }