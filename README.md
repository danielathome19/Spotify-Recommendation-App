# My Spotify Demo

This project is a local demonstration of a music recommendation system using Flask, FastAPI, SQLite, and a CSV dataset of Spotify-like tracks. The application allows users to register, log in, view tracks, listen to tracks, and get track recommendations based on their listening history.

## Project Structure

```
my_spotify_demo/
├── dataset.csv
├── init_db.py
├── db.sqlite
├── app.py
├── recommend_api.py
├── requirements.txt
└── templates/
    ├── base.html
    ├── index.html
    ├── register.html
    ├── login.html
    ├── tracks.html
    └── recommend.html
```

- **dataset.csv**: Contains the track metadata (Spotify-like columns).
- **init_db.py**: Script to create the SQLite schema and populate `db.sqlite`.
- **db.sqlite**: The local SQLite database file (generated after initialization).
- **app.py**: Main Flask application (routes for registration, login, track listing, etc.).
- **recommend_api.py**: FastAPI microservice for recommendation logic.
- **requirements.txt**: List of required dependencies.
- **templates/**: HTML (Jinja2) templates rendered by Flask routes.

## Environment & Dependencies

1. Create a virtual environment (recommended).
2. Install packages (`Flask`, `FastAPI`, `Uvicorn`, `pandas`, `bcrypt`, etc.).
3. Record these in `requirements.txt`.

## Database Schema & Initialization (init_db.py)

### Tables

Create three tables:
1. **users**: user accounts (`user_id`, `username`, `password_hash`, etc.).
2. **listening_history**: logs user-track interactions.
3. **tracks**: stores Spotify-like attributes (`track_id`, `artists`, etc.).

### Script Outline

`init_db.py` should define:

- `create_tables(conn)`: 
  - Drops existing tables (optional).
  - Creates `users`, `listening_history`, `tracks`.
- `populate_tracks(conn, csv_path)`: 
  - Reads `dataset.csv` via pandas or Python `csv`.
  - Inserts rows into `tracks`.
- `main()`:
  - Connects to `db.sqlite`.
  - Calls `create_tables(conn)`.
  - Calls `populate_tracks(conn, 'dataset.csv')`.
  - Closes the connection.

Run it once to initialize and populate `db.sqlite`.

## Flask Application (app.py)

**Flask** will handle:
- User Registration
- User Login
- Track Listing
- “Listening” to tracks (record in `listening_history`)
- Invoking the FastAPI endpoint for recommendations

### Database Connection
- `get_db_connection()` returns a connection to `db.sqlite`.

### User Routes
- `register()`: displays a form, inserts a new user into `users` with a hashed password.
- `login()`: checks credentials, sets session `user_id`.
- `logout()`: clears the session data.

### Track Management
- `tracks()`: lists tracks (e.g., top 50 from the DB).
- `listen(track_id)`: inserts a record into `listening_history`.

### Recommendation
- `recommend()`: calls the FastAPI (`/recommend/<user_id>`) to fetch recommended tracks.

## FastAPI Microservice (recommend_api.py)

A small recommendation service:

- `get_db_connection()`: connects to the same `db.sqlite`.
- `recommend_for_user(user_id: int, k: int = 5)`: 
  - `GET /recommend/{user_id}` 
  - Checks user listening history. 
  - Returns a JSON list of track IDs (random or dummy logic for demonstration).

## Templates (HTML)

Use **Jinja2** to create:

- `base.html`: base layout with navigation.
- `index.html`: homepage with links to login/register.
- `register.html`: registration form.
- `login.html`: login form.
- `tracks.html`: lists tracks, each has a “Listen” link.
- `recommend.html`: displays recommended track IDs from FastAPI.

## Running the Demo

1. **Initialize the DB**: 
  ```
  python init_db.py
  ```
2. **Start FastAPI**: 
  ```
  uvicorn recommend_api:app --port 8000 --reload
  ```
3. **Start Flask**: 
  ```
  python app.py
  ```
  Flask runs at `http://127.0.0.1:5000`.
4. **Test Flow**:
  - Go to `http://127.0.0.1:5000`.
  - Register & Login.
  - View tracks, click “Listen” to record a play.
  - Click “Get Recommendations” to call the FastAPI service.

## Enhancements

- **Security**: Consider `Flask-Login`, stronger secrets, etc.
- **Real ML Model**: Replace dummy logic in `recommend_for_user` with collaborative filtering or content-based methods.
- **Partial Fit**: Update the model incrementally as new data arrives.
- **UI**: Improve the front end for a more polished user experience.

## Summary of Methods to Implement

**init_db.py**
- `create_tables(conn)`
- `populate_tracks(conn, csv_path)`
- `main()`

**app.py**
- `get_db_connection()`
- `register()`
- `login()`
- `logout()`
- `tracks()`
- `listen(track_id)`
- `recommend()`

**recommend_api.py**
- `get_db_connection()`
- `recommend_for_user(user_id, k=5)`

These steps will create a **local demo** showing how Flask can manage user sessions and track listing, while FastAPI handles recommendation requests, all using a `SQLite` database and a `dataset.csv` of Spotify-like tracks.
