import pickle
import random
import sqlite3
import pandas as pd
from surprise import Dataset, Reader, KNNBasic


def get_db_connection():
    conn = sqlite3.connect('db.sqlite')
    conn.row_factory = sqlite3.Row
    return conn


def fit_model(data):
    reader = Reader(rating_scale=(0, 1))
    df = pd.DataFrame(data, columns=['user_id', 'track_id', 'rating'])
    dataset = Dataset.load_from_df(df, reader)
    trainset = dataset.build_full_trainset()

    model = KNNBasic()
    model.fit(trainset)

    with open('recommendation_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    pass


def train_initial_model():
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1) Check for any real listening data
    cursor.execute("SELECT COUNT(*) as cnt FROM listening_history")
    listens_count = cursor.fetchone()["cnt"]

    # 2) If no real data, fallback to top-10 approach
    if listens_count == 0:
        print("No user listening data found. Using fallback approach with top-10 popular tracks.")

        # a) Get the top 10 tracks by popularity
        cursor.execute("""
            SELECT track_id
            FROM tracks
            ORDER BY popularity DESC
            LIMIT 10
        """)
        top_tracks = [row["track_id"] for row in cursor.fetchall()]

        # b) Create a synthetic user, e.g. user_id = 999999
        synthetic_user_id = 999999
        data = []

        # "Like" these top 10 tracks with rating=1
        for tid in top_tracks:
            data.append((str(synthetic_user_id), tid, 1.0))

        # c) Add negative samples for the synthetic user
        cursor.execute("SELECT track_id FROM tracks")
        all_tracks = [row["track_id"] for row in cursor.fetchall()]
        not_listened = list(set(all_tracks) - set(top_tracks))
        negative_samples = random.sample(not_listened, min(len(not_listened), 10))
        for tid in negative_samples:
            data.append((str(synthetic_user_id), tid, 0.0))

        # d) Build a Surprise dataset from synthetic data and train the model
        fit_model(data)
        print("Fallback model trained on top-10 synthetic data. Stored as recommendation_model.pkl.")

    else:
        # 3) Real data approach
        print(f"Found {listens_count} listening records. Training with real user data.")

        # a) Fetch user_id, track_id, rating
        cursor.execute("SELECT user_id, track_id, rating FROM listening_history")
        all_listens = cursor.fetchall()

        # b) Build user->tracks mapping, separating positive vs. negative from DB
        user_tracks_positive = {}
        user_tracks_negative = {}
        for row in all_listens:
            uid = row["user_id"]
            tid = row["track_id"]
            r   = row["rating"]
            # Separate by rating
            if r >= 0.5:  # treat >= 0.5 as positive sentiment
                user_tracks_positive.setdefault(uid, set()).add(tid)
            else:
                user_tracks_negative.setdefault(uid, set()).add(tid)

        # c) Get all tracks for random sampling
        cursor.execute("SELECT track_id FROM tracks")
        all_tracks = [row["track_id"] for row in cursor.fetchall()]

        data = []

        # d) For each user, ensure at least 10 negative samples
        for uid in set(list(user_tracks_positive.keys()) + list(user_tracks_negative.keys())):
            # Add all the positives from DB (rating=1.0 or whatever was stored)
            if uid in user_tracks_positive:
                for tid in user_tracks_positive[uid]:
                    data.append((str(uid), tid, 1.0))

            # Add the negative from DB (rating=0.0) if user_tracks_negative has them
            existing_negatives = user_tracks_negative.get(uid, set())
            for tid in existing_negatives:
                data.append((str(uid), tid, 0.0))

            # Ensure user has >= 10 negative samples total
            negative_count = len(existing_negatives)
            if negative_count < 10:
                # Randomly pick additional tracks that are not already listened (nor in existing_negatives)
                pos_tracks = user_tracks_positive.get(uid, set())
                all_negative_candidates = set(all_tracks) - pos_tracks - existing_negatives

                needed = 10 - negative_count
                sample_size = min(needed, len(all_negative_candidates))
                additional_negs = random.sample(list(all_negative_candidates), sample_size)
                for tid in additional_negs:
                    data.append((str(uid), tid, 0.0))

        # e) Build a Surprise dataset from synthetic data and train the model
        fit_model(data)
        print("Model trained with real data. Stored as recommendation_model.pkl.")

    conn.close()


if __name__ == "__main__":
    train_initial_model()
