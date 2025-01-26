import sys
import time
import pickle
import sqlite3
import threading
import subprocess
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from tensorflow.keras.models import Model

app = FastAPI()

# Global model + lock to protect the
# model from being accessed concurrently
model: Model
mappings: dict
user_map: dict
track_map: dict
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
    global model, mappings, user_map, track_map
    with model_lock:
        model = tf.keras.models.load_model('recommendation_model_2.h5')
        with open('mappings.pkl', 'rb') as f:
            mappings = pickle.load(f)
            user_map = mappings['user_map']
            track_map = mappings['track_map']
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
        subprocess.run([python_exe, "train_model_2.py"])

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
    # Acquire the model lock while reading model
    with model_lock:
        current_model = model


    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT track_id, artists, track_name FROM tracks")
    all_tracks = cursor.fetchall()
    conn.close()

    if user_id not in user_map:
        print(f"User {user_id} not found in user_map.")
        return {"recommended_tracks": []}

    # Debug user and track mapping
    print("User Map:", user_map)
    print("Track Map:", track_map)

    # Debug: Ensure np is correct
    print("Type of np:", type(np))
    if not isinstance(track_map, dict):
        print("Error: track_map is not a dictionary.")
        return {"error": "Invalid track map."}

    # Ensure user exists in mapping
    if user_id not in user_map:
        return {"recommended_tracks": []}

    # Prepare input for prediction
    user_idx = np.array([user_map[user_id]] * len(track_map))  # Repeat user index for all tracks
    track_idxs = np.array(list(track_map.values()))            # All track indices

    print("User Index Shape:", user_idx.shape)
    print("Track Indices Shape:", track_idxs.shape)

    # Predict scores
    scores = current_model.predict([user_idx, track_idxs], batch_size=128).flatten()

    # Get Top-K Recommendations
    top_k_indices = scores.argsort()[-k:][::-1]
    top_k = [
        {
            "artists": all_tracks[i]['artists'],
            "track_name": all_tracks[i]['track_name'],
            "score": float(scores[i]),  # Convert numpy.float32 to Python float
        }
        for i in top_k_indices
    ]

    return {"recommended_tracks": top_k}
