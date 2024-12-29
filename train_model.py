import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import pickle

def get_db_connection():
    conn = sqlite3.connect('db.sqlite')
    conn.row_factory = sqlite3.Row
    return conn

def train_initial_model():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Fetch listening history data
    cursor.execute("SELECT user_id, track_id FROM listening_history")
    listening_history = cursor.fetchall()
    
    # Fetch track attributes
    cursor.execute("SELECT * FROM tracks")
    tracks = cursor.fetchall()
    
    # Create a DataFrame for listening history
    history_df = pd.DataFrame(listening_history, columns=['user_id', 'track_id'])
    
    # Create a DataFrame for tracks
    tracks_df = pd.DataFrame(tracks, columns=[
        'track_id', 'artists', 'album_name', 'track_name', 'popularity', 'duration_ms', 'explicit',
        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'track_genre'
    ])
    
    # Merge listening history with track attributes
    merged_df = pd.merge(history_df, tracks_df, on='track_id')
    
    # Select features for the model
    features = merged_df[['popularity', 'duration_ms', 'danceability', 'energy', 'key', 'loudness',
                          'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                          'valence', 'tempo', 'time_signature']]
    
    # Train-test split
    X_train, X_test = train_test_split(features, test_size=0.2, random_state=42)
    
    # Train a NearestNeighbors model
    model = NearestNeighbors(n_neighbors=5, algorithm='auto')
    model.fit(X_train)
    
    # Save the trained model to a file
    with open('recommendation_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    conn.close()

if __name__ == "__main__":
    train_initial_model()
