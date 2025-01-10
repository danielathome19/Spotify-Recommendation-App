import json
import bcrypt
import logging
import sqlite3
from faker import Faker

fake = Faker()


def track_exists(conn, track_id):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tracks WHERE track_id = ?", (track_id,))
    return cursor.fetchone() is not None


def populate_users(conn):
    cursor = conn.cursor()

    # Load the JSON data from "mpd.slice.0-999.json", the first slice of the (Spotify) Million Playlist Dataset
    with open('mpd.slice.0-999.json') as f:
        data = json.load(f)

    # Build a faker user for each playlist in the dataset as (listening_history [list of str track_ids])
    # Every song in the playlist (that exists in the tracks table) is added to the user's listening history
    all_playlists = []
    for playlist in data['playlists']:
        current_playlist = []
        for track in playlist['tracks']:
            t_uri = track['track_uri'].replace('spotify:track:', '')
            if track_exists(conn, t_uri):
                current_playlist.append(t_uri)
        if len(current_playlist) > 0:
            all_playlists.append(current_playlist)
        else:
            pass  # ("Playlist has no valid tracks: ", playlist['name'])

    print("Total playlists: ", len(all_playlists))
    logging.basicConfig(filename='logs/fake_users.log', level=logging.INFO)
    for i, playlist in enumerate(all_playlists):
        username = fake.user_name() + str(i)
        password = fake.password()
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
        logging.info(f"User {username} created with password {password}")

        # Get the user_id of the user we just created
        cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        user_id = cursor.fetchone()[0]

        # Insert each track in the playlist into the listening_history table
        for track_id in playlist:
            cursor.execute("INSERT INTO listening_history (user_id, track_id, rating, listened_at) VALUES (?, ?, ?, ?)",
                           (user_id, track_id, 1, fake.date_time_this_year()))
        print(f"Inserted playlist {i + 1} into the database.")

    conn.commit()
    conn.close()


def main():
    conn = sqlite3.connect('db.sqlite')
    populate_users(conn)
    conn.close()


if __name__ == "__main__":
    main()