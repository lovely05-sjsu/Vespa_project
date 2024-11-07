import pandas as pd
import json

def collapse_genres(genre_json):    #function to process the genres field if it's in JSON format
    genres = []
    try:
        genre_data = json.loads(genre_json)
        for genre in genre_data:
            genres.append(genre.get("name", ""))
    except (TypeError, json.JSONDecodeError):
        pass  # Handle missing or malformed JSON
    return " ".join(sorted(genres))

# Process the TMDB movie dataset for Vespa
def process_tmdb_csv(input_file, output_file):
    movies = pd.read_csv(input_file)      
    if 'genres' in movies.columns:
        movies['genres_name'] = movies['genres'].apply(collapse_genres)
    else:
        movies['genres_name'] = ''
    
    # Fill missing values
    for column in ['original_title', 'overview', 'genres_name']:
        movies[column] = movies[column].fillna('')
    
    movies["text"] = movies["overview"] + " " + movies["genres_name"]   # Combine overview and genres into the "text" field
    
    # Select and rename columns for Vespa compatibility
    movies = movies[['id', 'original_title', 'text']]
    movies.rename(columns={'original_title': 'title', 'id': 'doc_id'}, inplace=True)
    
    # Create JSON structure for Vespa's "put" and "fields" requirements
    movies['fields'] = movies.apply(lambda row: row.to_dict(), axis=1)
    movies['put'] = movies['doc_id'].apply(lambda x: f"id:hybrid-search:doc::{x}")
    
    # Select only the necessary columns and save to JSONL
    df_result = movies[['put', 'fields']]
    print(df_result.head())
    df_result.to_json(output_file, orient='records', lines=True)
    print(f"Processed data saved to {output_file}")

process_tmdb_csv ("movies_tmdb_popular.csv", "clean_tmdb_data.jsonl")  
