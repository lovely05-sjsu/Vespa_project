import pandas as pd
from vespa.application import Vespa
from vespa.io import VespaQueryResponse, VespaResponse
import numpy as np


# Function to display the search results in DataFrame format
def display_hits_as_df(response: VespaQueryResponse, fields):
    records = []
    for hit in response.hits:
        record = {}
        for field in fields:
            value = hit["fields"].get(field)
            if isinstance(value, dict) and 'values' in value:  # Handle tensor data
                record[field] = np.array(value['values']).tolist()  # Convert tensor to list
            else:
                record[field] = value
        records.append(record)
    return pd.DataFrame(records)


# Keyword search
def keyword_search(app, search_query):
    query = {
        "yql": "select * from sources * where userQuery() limit 5",  # A simple keyword search
        "query": search_query,
        "ranking": "bm25",  # Ranking function for keyword-based search
    }
    response = app.query(query)
    return display_hits_as_df(response, ["doc_id", "title", "text"])


# Semantic search using nearest neighbor (requires embeddings)
def semantic_search(app, query):
    query = {
        "yql": "select * from sources * where ({targetHits:100}nearestNeighbor(embedding,e)) limit 5",  # Semantic search
        "query": query,
        "ranking": "semantic",  # Ranking profile for semantic search
        "input.query(e)": "embed(@query)"  # Embedding model for the query
    }
    response = app.query(query)
    return display_hits_as_df(response, ["doc_id", "title", "text"])


# Fetch embedding for a specific document by its doc_id
def get_embedding(app, doc_id):
    query = {
        "yql" : f"select doc_id, title, text, embedding from content.doc where doc_id contains '{doc_id}'",
        "hits": 1  # We expect only one document
    }
    result = app.query(query)
    
    if result.hits:
        return result.hits[0]
    return None


# Recommendation search based on a document's embedding
def query_movies_by_embedding(app, embedding_vector):
    query = {
        'hits': 5,
        'yql': 'select * from content.doc where ({targetHits:5}nearestNeighbor(embedding, user_embedding))',
        'ranking.features.query(user_embedding)': str(embedding_vector),
        'ranking.profile': 'recommendation'
    }
    response = app.query(query)
    return display_hits_as_df(response, ["doc_id", "title", "text"])


# Connect to Vespa
app = Vespa(url="http://localhost", port=8080)

# Perform Keyword Search
query = "Harry Potter and the Half-Blood Prince"
df_keyword = keyword_search(app, query)
print("Keyword Search Results:")
print(df_keyword.head())

# Perform Semantic Search
df_semantic = semantic_search(app, query)
print("\nSemantic Search Results:")
print(df_semantic.head())

# Get embedding for a specific doc_id (Example doc_id: 878)
embedding_data = get_embedding(app, "878")
if embedding_data:
    print("\nEmbedding for doc_id 878:", embedding_data["fields"].get("embedding"))

    # Perform Recommendation Search based on this embedding
    df_recommendation = query_movies_by_embedding(app, embedding_data["fields"].get("embedding"))
    print("\nRecommendation Search Results based on doc_id 878:")
    print(df_recommendation.head())
else:
    print("\nNo embedding found for doc_id 878")
