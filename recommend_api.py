import sys
import time
import pickle
import sqlite3
import threading
import subprocess
from fastapi import FastAPI
from surprise import KNNBasic

app = FastAPI()

# Global model + lock to protect the
# model from being accessed concurrently
model: KNNBasic
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
    global model
    with model_lock:
        with open('recommendation_model.pkl', 'rb') as f:
            model = pickle.load(f)
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
        time.sleep(60)  # TODO: change back to 3600

        # Run the training script (blocking call)
        print("[Scheduler] Starting model retraining...")
        subprocess.run([python_exe, "train_model.py"])

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

    cursor.execute("SELECT track_id FROM listening_history WHERE user_id = ?", (user_id,))
    listening_history = [row['track_id'] for row in cursor.fetchall()]

    cursor.execute("SELECT track_id, artists, track_name FROM tracks")
    all_tracks = cursor.fetchall()
    conn.close()

    # Score each track that the user hasn't listened to
    recommendations = []
    for row in all_tracks:
        tid = row['track_id']
        if tid not in listening_history:
            est = current_model.predict(str(user_id), tid).est
            recommendations.append((row['artists'], row['track_name'], est))

    # Sort by rating estimate
    recommendations.sort(key=lambda x: x[2], reverse=True)
    top_k = recommendations[:k]

    # Format result
    result = []
    for item in top_k:
        result.append({
            "artists": item[0],
            "track_name": item[1],
            "score": item[2]
        })
    return {"recommended_tracks": result}
