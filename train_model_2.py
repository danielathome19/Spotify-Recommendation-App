import os
import sys
import pickle
import sqlite3
import logging
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout

# Connect to GPU if possible
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TORCH_USE_CUDA_DSA'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print("Connected to", gpu)
    tf.config.experimental.set_memory_growth(gpu, True)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_db_connection():
    conn = sqlite3.connect('db.sqlite')
    conn.row_factory = sqlite3.Row
    return conn


def prepare_data():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch user_id, track_id, and rating
    cursor.execute("SELECT user_id, track_id, rating FROM listening_history")
    data = cursor.fetchall()

    # Encode user and track IDs as integers
    df = pd.DataFrame(data, columns=['user_id', 'track_id', 'rating'])
    user_map = {u: i for i, u in enumerate(df['user_id'].unique())}
    track_map = {t: i for i, t in enumerate(df['track_id'].unique())}
    df['user_idx'] = df['user_id'].map(user_map)
    df['track_idx'] = df['track_id'].map(track_map)

    conn.close()
    return df, len(user_map), len(track_map), user_map, track_map


def build_model(num_users, num_tracks, embed_dim=50, metadata_dim=0):
    # User and Track Embeddings
    user_input = Input(shape=(1,), name='user_input')
    user_embed = Embedding(num_users, embed_dim, name='user_embedding')(user_input)
    user_vec = Flatten()(user_embed)

    track_input = Input(shape=(1,), name='track_input')
    track_embed = Embedding(num_tracks, embed_dim, name='track_embedding')(track_input)
    track_vec = Flatten()(track_embed)

    # Optional Metadata Input
    if metadata_dim > 0:
        metadata_input = Input(shape=(metadata_dim,), name='metadata_input')
        metadata_vec = Dense(32, activation='relu')(metadata_input)
        x = Concatenate()([user_vec, track_vec, metadata_vec])
    else:
        x = Concatenate()([user_vec, track_vec])

    # Fully Connected Layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(1, activation='sigmoid', name='output')(x)

    # Compile Model
    inputs = [user_input, track_input] if metadata_dim == 0 else [user_input, track_input, metadata_input]
    model = Model(inputs, x)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_initial_model():
    df, num_users, num_tracks, user_map, track_map = prepare_data()

    # Train-Test Split
    train, val = train_test_split(df, test_size=0.2, random_state=42)

    # Prepare Inputs
    x_train = [train['user_idx'], train['track_idx']]
    x_val = [val['user_idx'], val['track_idx']]
    y_train = train['rating']
    y_val = val['rating']

    # Build and Train Model
    model = build_model(num_users, num_tracks)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)

    # Save Model and Metadata
    model.save('recommendation_model_2.h5')
    with open('mappings.pkl', 'wb') as f:
        pickle.dump({'user_map': user_map, 'track_map': track_map}, f)
        # pickle.dump({'user_map': df['user_id'].map(user_map), 'track_map': df['track_id'].map(track_map)}, f)


if __name__ == "__main__":
    train_initial_model()
