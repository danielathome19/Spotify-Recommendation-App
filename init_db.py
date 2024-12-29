import sqlite3
import pandas as pd

def create_tables(conn):
    cursor = conn.cursor()
    
    # Drop existing tables (optional)
    cursor.execute("DROP TABLE IF EXISTS users")
    cursor.execute("DROP TABLE IF EXISTS listening_history")
    cursor.execute("DROP TABLE IF EXISTS tracks")
    
    # Create users table
    cursor.execute("""
    CREATE TABLE users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL
    )
    """)
    
    # Create listening_history table
    cursor.execute("""
    CREATE TABLE listening_history (
        history_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        track_id TEXT,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
    """)
    
    # Create tracks table
    cursor.execute("""
    CREATE TABLE tracks (
        track_id TEXT PRIMARY KEY,
        artists TEXT,
        album_name TEXT,
        track_name TEXT,
        popularity INTEGER,
        duration_ms INTEGER,
        explicit BOOLEAN,
        danceability REAL,
        energy REAL,
        key INTEGER,
        loudness REAL,
        mode INTEGER,
        speechiness REAL,
        acousticness REAL,
        instrumentalness REAL,
        liveness REAL,
        valence REAL,
        tempo REAL,
        time_signature INTEGER,
        track_genre TEXT
    )
    """)
    
    conn.commit()

def populate_tracks(conn, csv_path):
    df = pd.read_csv(csv_path)
    df.to_sql('tracks', conn, if_exists='append', index=False)

def main():
    conn = sqlite3.connect('db.sqlite')
    create_tables(conn)
    populate_tracks(conn, 'dataset.csv')
    conn.close()

if __name__ == "__main__":
    main()
