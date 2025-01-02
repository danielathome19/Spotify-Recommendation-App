from fastapi import FastAPI
import sqlite3
import pickle
import pandas as pd
from surprise import Dataset, Reader, KNNBasic

app = FastAPI()

def get_db_connection():
    conn = sqlite3.connect('db.sqlite')
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/recommend/{user_id}")
def recommend_for_user(user_id: int, k: int = 5):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Fetch user listening history
    cursor.execute("SELECT track_id FROM listening_history WHERE user_id = ?", (user_id,))
    listening_history = cursor.fetchall()
    
    # Load the trained model
    with open('recommendation_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Get the list of all track IDs
    cursor.execute("SELECT track_id FROM tracks")
    all_tracks = cursor.fetchall()
    all_track_ids = [track['track_id'] for track in all_tracks]
    
    # Generate recommendations
    recommended_tracks = []
    for track_id in all_track_ids:
        if track_id not in [track['track_id'] for track in listening_history]:
            # est = model.predict(user_id, track_id).est  # Est is the estimated rating
            est = model.predict(str(user_id), track_id).est
            track_info = cursor.execute("SELECT artists, track_name FROM tracks WHERE track_id = ?", (track_id,)).fetchone()
            recommended_tracks.append((track_info, est))
    
    # Sort the recommendations by estimated rating
    recommended_tracks.sort(key=lambda x: x[1], reverse=True)
    recommended_tracks = [track[0] for track in recommended_tracks[:k]]
    
    conn.close()
    return {"recommended_tracks": recommended_tracks}
