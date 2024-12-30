import sqlite3
import pandas as pd
from surprise import Dataset, Reader, KNNBasic
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
    
    # Prepare the data for the Surprise library
    data = [(row['user_id'], row['track_id'], 1) for row in listening_history]
    reader = Reader(rating_scale=(0, 1))
    dataset = Dataset.load_from_df(pd.DataFrame(data, columns=['user_id', 'track_id', 'rating']), reader)
    trainset = dataset.build_full_trainset()
    
    # Train a collaborative filtering model using the Surprise library
    model = KNNBasic()
    model.fit(trainset)
    
    # Save the trained model to a file
    with open('recommendation_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    conn.close()

if __name__ == "__main__":
    train_initial_model()
