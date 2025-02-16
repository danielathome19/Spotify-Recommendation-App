import bcrypt
import sqlite3
import requests
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = 'supersecretkey'


def get_db_connection():
    conn = sqlite3.connect('db.sqlite')
    conn.row_factory = sqlite3.Row
    return conn


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
        conn.commit()
        conn.close()

        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash']):
            session['user_id'] = user['user_id']
            session['username'] = user['username']  # TODO: keep or remove
            return redirect(url_for('tracks'))
        else:
            return "Invalid credentials"
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))


@app.route('/tracks')
def tracks():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tracks LIMIT 500")  # TODO: paginate, add dislike
    tracks = cursor.fetchall()
    conn.close()
    return render_template('tracks.html', tracks=tracks)


@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT listening_history.listened_at, tracks.track_id, tracks.artists, tracks.track_name
        FROM listening_history
        JOIN tracks ON listening_history.track_id = tracks.track_id
        WHERE listening_history.user_id = ?
        ORDER BY listening_history.listened_at DESC
    """, (user_id,))
    history = cursor.fetchall()
    conn.close()
    return render_template('history.html', history=history)


@app.route('/listen/<track_id>')
def listen(track_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO listening_history (user_id, track_id, listened_at) VALUES (?, ?, ?)", (user_id, track_id, datetime.now()))
    conn.commit()
    conn.close()
    return redirect(url_for('tracks'))


@app.route('/recommend')
def recommend():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    response = requests.get(f'http://127.0.0.1:8000/recommend/{user_id}')
    recommended_tracks = response.json().get('recommended_tracks', [])
    return render_template('recommend.html', recommended_tracks=recommended_tracks)


if __name__ == "__main__":
    app.run(debug=True)
