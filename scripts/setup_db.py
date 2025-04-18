# scripts/setup_database.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2

def setup_minneo_database():
    """Create the minneo_db database if it doesn't exist."""
    # Connect to default postgres database
    try:
        # Connect to the default 'postgres' database first
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="postgres",
            user="ianparr"
        )
        conn.autocommit = True  # Need this for creating databases
        cursor = conn.cursor()

        # Check if database already exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname='minneo_db'")
        exists = cursor.fetchone()

        if not exists:
            print("Creating minneo_db database...")
            cursor.execute("CREATE DATABASE minneo_db")
            print("Database created successfully!")
        else:
            print("Database minneo_db already exists.")

        cursor.close()
        conn.close()

        print("Database setup complete!")
    except Exception as e:
        print(f"Error setting up database: {e}")

if __name__ == "__main__":
    setup_minneo_database()