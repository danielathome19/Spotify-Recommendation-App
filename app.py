from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import bcrypt

app = Flask(__name__)
app.secret_key = 'your_secret_key'

def get_db_connection():
    conn = sqlite3.connect('db.sqlite')
    conn.row_factory = sqlite3.Row
    return conn

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
            return redirect(url_for('tracks'))
        else:
            return 'Invalid credentials'
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/tracks')
def tracks():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tracks LIMIT 50")
    tracks = cursor.fetchall()
    conn.close()
    return render_template('tracks.html', tracks=tracks)

@app.route('/listen/<track_id>')
def listen(track_id):
    if 'user_id' in session:
        user_id = session['user_id']
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO listening_history (user_id, track_id) VALUES (?, ?)", (user_id, track_id))
        conn.commit()
        conn.close()
        return redirect(url_for('tracks'))
    return redirect(url_for('login'))

@app.route('/recommend')
def recommend():
    if 'user_id' in session:
        user_id = session['user_id']
        # Call FastAPI endpoint for recommendations
        # For now, we'll just return a placeholder
        recommended_tracks = ["track1", "track2", "track3"]
        return render_template('recommend.html', recommended_tracks=recommended_tracks)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
