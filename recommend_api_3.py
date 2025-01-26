import sys
import time
import pickle
import sqlite3
import threading
import subprocess
import numpy as np
from fastapi import FastAPI
# from surprise import AlgoBase
import tensorflow as tf
from tensorflow.keras.models import Model

app = FastAPI()

# Global model + lock to protect the
# model from being accessed concurrently
model: Model
mappings: dict
user_map: dict
track_map: dict
genre_map: dict
model_lock = threading.Lock()
python_exe = sys.executable  # Get current Python executable path


def get_db_connection():
    conn = sqlite3.connect('db.sqlite')
    conn.row_factory = sqlite3.Row
    return conn


def load_model():
    """
    Safely reload the model from recommendation_model.pkl
    using a lock to avoid concurrency issues.
    """
    global model, mappings, user_map, track_map, genre_map
    with model_lock:
        model = tf.keras.models.load_model('recommendation_model_3.h5')
        with open('mappings_large.pkl', 'rb') as f:
            mappings = pickle.load(f)
            user_map = mappings['user_map']
            track_map = mappings['track_map']
            genre_map = mappings['genre_map']
            print("[Model] Reloaded model from disk.")


def background_scheduler():
    """
    Runs in a separate thread.
    Every hour:
      1) Calls `train_model.py` to retrain the model
      2) Reloads the model into memory
    """
    while True:
        # Sleep 1 hour (3600 seconds)
        time.sleep(3600)

        # Run the training script (blocking call)
        print("[Scheduler] Starting model retraining...")
        subprocess.run([python_exe, "train_model_3.py"])

        # Reload the model after training completes
        load_model()
        print("[Scheduler] Model reloaded after retraining.")


# Start the scheduler thread on startup
@app.on_event("startup")
def on_startup():
    # Load initial model once at startup
    load_model()

    # Start the scheduler thread (daemon so it won't block shutdown)
    scheduler_thread = threading.Thread(target=background_scheduler, daemon=True)
    scheduler_thread.start()


@app.get("/recommend/{user_id}")
def recommend_for_user(user_id: int, k: int = 5):
    global user_map, track_map, genre_map
    # Acquire the model lock while reading model
    with model_lock:
        current_model = model

    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch all track metadata for predictions
    cursor.execute("SELECT track_id, artists, track_name, track_genre, popularity FROM tracks")
    all_tracks = cursor.fetchall()  # Example: [{track_id: ..., artists: ..., ...}]
    conn.close()

    # Score each track that the user hasn't listened to
    # recommendations = []
    # for row in all_tracks:
    #     tid = row['track_id']
    #     if tid not in listening_history:
    #         est = current_model.predict(str(user_id), tid).est
    #         recommendations.append((row['artists'], row['track_name'], est))
    #
    # # Sort by rating estimate
    # recommendations.sort(key=lambda x: x[2], reverse=True)
    # top_k = recommendations[:k]

    # Return fallback if the user is not in the mapping
    if user_id not in user_map:
        # Recommend top-k popular tracks if user has no data
        top_tracks = sorted(all_tracks, key=lambda x: x['popularity'], reverse=True)[:k]
        return {
            "recommended_tracks": [
                {"artists": track['artists'], "track_name": track['track_name'], "score": float(track['popularity'])}
                for track in top_tracks
            ]
        }

    # Construct input arrays
    user_idx = np.array([user_map[user_id]] * len(all_tracks), dtype=np.int32)
    track_idxs = np.array([track_map.get(track['track_id'], 0) for track in all_tracks], dtype=np.int32)
    genre_idxs = np.array([genre_map.get(track['track_genre'], 0) for track in all_tracks], dtype=np.int32)
    popularity = np.array([track['popularity'] if track['popularity'] is not None else 0 for track in all_tracks],
                          dtype=np.float32)

    # Check for consistent shapes
    assert len(user_idx) == len(track_idxs) == len(genre_idxs) == len(
        popularity), "Input arrays must have the same length"

    # Predict scores
    scores = current_model.predict([user_idx, track_idxs, genre_idxs, popularity], batch_size=128).flatten()

    # Predict scores
    #scores = current_model.predict([user_idx, track_idxs], batch_size=128).flatten()

    # Get Top-K Recommendations
    try:
        top_k_indices = scores.argsort()[-k:][::-1]
        # top_k = [(all_tracks[i]['artists'], all_tracks[i]['track_name'], scores[i]) for i in top_k_indices]
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        return {"error": str(e)}

    # # Normalize scores using Z-score normalization
    # if recommendations:
    #     scores = [item[2] for item in recommendations]  # Extract the raw scores
    #     mean = sum(scores) / len(scores)
    #     std = (sum((x - mean) ** 2 for x in scores) / len(scores)) ** 0.5 or 1  # Avoid division by zero
    #
    #     # Update recommendations with normalized scores
    #     recommendations = [(item[0], item[1], (item[2] - mean) / std) for item in recommendations]
    #
    # # Sort by normalized score and take the top-k
    # recommendations.sort(key=lambda x: x[2], reverse=True)
    # top_k = recommendations[:k]

    # Format result
    result = [
        {
            "artists": all_tracks[idx]['artists'],
            "track_name": all_tracks[idx]['track_name'],
            "score": float(scores[idx])
        }
        for idx in top_k_indices
    ]
    return {"recommended_tracks": result}
