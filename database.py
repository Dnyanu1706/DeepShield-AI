
import sqlite3
from pathlib import Path

# Database will be created in the main project folder
BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "database.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    connection = get_connection()

    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        connection.commit()
        print("Database initialized successfully!")
        print(f"Database location: {DATABASE_PATH}")

    finally:
        connection.close()


def save_prediction(filename, prediction, confidence):
    connection = get_connection()

    try:
        connection.execute(
            """
            INSERT INTO history (filename, prediction, confidence)
            VALUES (?, ?, ?)
            """,
            (filename, prediction, confidence),
        )
        connection.commit()

    finally:
        connection.close()


def get_history(limit=100):
    connection = get_connection()

    try:
        return connection.execute(
            """
            SELECT id, filename, prediction, confidence, created_at
            FROM history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    finally:
        connection.close()


if __name__ == "__main__":
    initialize_database()
