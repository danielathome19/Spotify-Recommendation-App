import os
import sys
import pickle
import logging
import sqlite3
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout

# Connect to GPU if possible
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TORCH_USE_CUDA_DSA'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
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

    # Fetch listening history with metadata
    cursor.execute("""
        SELECT lh.user_id, lh.track_id, lh.rating, t.track_genre, t.popularity
        FROM listening_history lh
        JOIN tracks t ON lh.track_id = t.track_id
    """)
    listening_data = cursor.fetchall()

    # Fetch all tracks for mapping
    cursor.execute("SELECT track_id, track_genre, popularity FROM tracks")
    all_tracks = cursor.fetchall()

    # Create mappings for user_id, track_id, and track_genre
    listening_df = pd.DataFrame(listening_data, columns=['user_id', 'track_id', 'rating', 'track_genre', 'popularity'])
    all_tracks_df = pd.DataFrame(all_tracks, columns=['track_id', 'track_genre', 'popularity'])

    user_map = {u: i for i, u in enumerate(listening_df['user_id'].unique())}
    track_map = {t: i for i, t in enumerate(all_tracks_df['track_id'].unique())}
    genre_map = {g: i for i, g in enumerate(all_tracks_df['track_genre'].unique())}

    # Add mapped indices to listening data
    listening_df['user_idx'] = listening_df['user_id'].map(user_map)
    listening_df['track_idx'] = listening_df['track_id'].map(track_map)
    listening_df['genre_idx'] = listening_df['track_genre'].map(genre_map)

    conn.close()

    # Return the listening data, all track metadata, and mappings
    return (
        listening_df,
        len(user_map),
        len(track_map),
        len(genre_map),
        user_map,
        track_map,
        genre_map
    )


def build_model(num_users, num_tracks, num_genres, embed_dim=50):
    # User and Track Embeddings
    user_input = Input(shape=(1,), name='user_input')
    user_embed = Embedding(num_users, embed_dim, name='user_embedding', embeddings_regularizer=l2(1e-6))(user_input)
    user_vec = Flatten()(user_embed)

    track_input = Input(shape=(1,), name='track_input')
    track_embed = Embedding(num_tracks, embed_dim, name='track_embedding', embeddings_regularizer=l2(1e-6))(track_input)
    track_vec = Flatten()(track_embed)

    # Genre Embedding
    genre_input = Input(shape=(1,), name='genre_input')
    genre_embed = Embedding(num_genres, embed_dim // 2, name='genre_embedding', embeddings_regularizer=l2(1e-6))(genre_input)
    genre_vec = Flatten()(genre_embed)

    # Metadata Input (e.g., popularity)
    popularity_input = Input(shape=(1,), name='popularity_input')
    popularity_vec = Dense(8, activation='relu')(popularity_input)

    # Concatenate All Features
    x = Concatenate()([user_vec, track_vec, genre_vec, popularity_vec])

    # Fully Connected Layers
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dense(1, activation='sigmoid', name='output')(x)

    # Compile Model
    model = Model(inputs=[user_input, track_input, genre_input, popularity_input], outputs=x)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_initial_model():
    df, num_users, num_tracks, num_genres, user_map, track_map, genre_map = prepare_data()

    # Negative Sampling
    negative_samples = []
    all_tracks = set(track_map.values())
    for user, group in df.groupby('user_idx'):
        pos_tracks = set(group['track_idx'])
        num_negatives = len(pos_tracks) * 2  # Twice as many negatives as positives
        neg_tracks = list(all_tracks - pos_tracks)
        sampled_negs = np.random.choice(neg_tracks, size=min(num_negatives, len(neg_tracks)), replace=False)
        for neg in sampled_negs:
            negative_samples.append([user, neg, 0, None, None])  # No metadata for negative samples

    negative_df = pd.DataFrame(negative_samples, columns=['user_idx', 'track_idx', 'rating', 'genre_idx', 'popularity'])
    full_df = pd.concat([df, negative_df], ignore_index=True)

    # Train-Test Split
    train, val = train_test_split(full_df, test_size=0.2, random_state=42)

    # Prepare Inputs
    x_train = [train['user_idx'], train['track_idx'], train['genre_idx'].fillna(0), train['popularity'].fillna(0)]
    x_val = [val['user_idx'], val['track_idx'], val['genre_idx'].fillna(0), val['popularity'].fillna(0)]
    y_train = train['rating']
    y_val = val['rating']

    # Build and Train Model
    model = build_model(num_users, num_tracks, num_genres)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=64, callbacks=[early_stopping])

    # Save Model and Metadata
    model.save('recommendation_model_3.h5')
    with open('mappings_large.pkl', 'wb') as f:
        pickle.dump({'user_map': user_map, 'track_map': track_map, 'genre_map': genre_map}, f)


if __name__ == "__main__":
    train_initial_model()
