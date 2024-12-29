from fastapi import FastAPI
import sqlite3

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
    
    # Dummy logic for recommendations (replace with real ML model)
    recommended_tracks = [track['track_id'] for track in listening_history][:k]
    
    conn.close()
    return {"recommended_tracks": recommended_tracks}
