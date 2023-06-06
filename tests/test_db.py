import pytest
import unittest.mock as mock
from db.tables import UserQueryTable, InsertExpection 
from db.client import get_client, get_ns

@pytest.fixture
def mock_client():
    return mock.create_autospec(get_client(None), instance=True)

@pytest.fixture
def mock_bigquery():
    return mock.create_autospec(get_ns(None))

def test_user_query_table_init(mock_client, mock_bigquery):
    mock_client.get_table.side_effect = Exception("Table not found")
    table = UserQueryTable('project_id', mock_client, mock_bigquery, 'dataset_id')
    
    assert table.client == mock_client
    mock_client.get_table.assert_called_once_with('project_id.dataset_id.user_queries')
    mock_client.create_table.assert_called_once()

def test_insert_query_success(mock_client, mock_bigquery):
    mock_client.insert_rows.return_value = []

    table = UserQueryTable('project_id', mock_client, mock_bigquery, 'dataset_id')
    query = "What's the weather like today?"

    query_id = table.insert_query(query)

    mock_client.get_table.assert_called_with('project_id.dataset_id.user_queries')
    assert mock_client.get_table.call_count == 2

def test_insert_query_failure(mock_client, mock_bigquery):
    mock_client.insert_rows.return_value = ['error']

    table = UserQueryTable('project_id', mock_client, mock_bigquery, 'dataset_id')
    query = "What's the weather like today?"

    with pytest.raises(InsertExpection):
        table.insert_query(query)

    mock_client.get_table.assert_called_with('project_id.dataset_id.user_queries')
    assert mock_client.get_table.call_count == 2
    mock_client.insert_rows.assert_called_once()

def test_upvote_query(mock_client, mock_bigquery):
    table = UserQueryTable('project_id', mock_client, mock_bigquery, 'dataset_id')
    query_id = '1234'

    table.upvote_query(query_id)

    mock_client.query.assert_called_once()

def test_downvote_query(mock_client, mock_bigquery):
    table = UserQueryTable('project_id', mock_client, mock_bigquery, 'dataset_id')
    query_id = '1234'

    table.downvote_query(query_id)

    mock_client.query.assert_called_once()