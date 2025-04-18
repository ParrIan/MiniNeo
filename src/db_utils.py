# src/db_utils.py
import psycopg2
from sqlalchemy import create_engine
from src.config import DB_CONFIG

def get_connection():
    """Get a raw psycopg2 connection to the database."""
    conn_params = {k: v for k, v in DB_CONFIG.items() if k != 'password' or v}

    try:
        conn = psycopg2.connect(**conn_params)
        return conn
    except psycopg2.OperationalError as e:
        print(f"Could not connect to PostgreSQL: {e}")
        print("Check if PostgreSQL is running and if the credentials are correct.")
        raise

def get_engine():
    """Get a SQLAlchemy engine."""
    user = DB_CONFIG['user']
    password = DB_CONFIG['password']
    host = DB_CONFIG['host']
    port = DB_CONFIG['port']
    database = DB_CONFIG['database']

    if password:
        connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    else:
        connection_string = f"postgresql://{user}@{host}:{port}/{database}"

    return create_engine(connection_string)

def execute_query(query, fetch=True):
    """Execute a query and return results."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(query)

    if fetch:
        results = cursor.fetchall()
    else:
        results = None
        conn.commit()

    cursor.close()
    conn.close()
    return results

def force_query_plan(query, plan_hints):
    """Execute a query with specific plan hints."""
    # This will need implementation based on PostgreSQL's hint syntax
    # or other plan forcing mechanisms
    modified_query = f"/*+ {plan_hints} */ {query}"
    return execute_query(modified_query)