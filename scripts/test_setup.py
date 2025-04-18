# scripts/test_setup.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from src.db_utils import execute_query, get_connection

def test_pytorch():
    # Check if PyTorch is working
    x = torch.rand(5, 3)
    print("PyTorch tensor:\n", x)
    if torch.cuda.is_available():
        print("CUDA is available. GPU will be used.")
    else:
        print("CUDA is not available. CPU will be used.")

def test_db_connection():
    # Test database connection
    try:
        conn = get_connection()
        print("Database connection successful!")
        conn.close()
    except Exception as e:
        print(f"Database connection failed: {e}")

    # Test query execution
    try:
        results = execute_query("SELECT version();")
        print(f"PostgreSQL version: {results[0][0]}")
    except Exception as e:
        print(f"Query execution failed: {e}")

if __name__ == "__main__":
    print("=== Testing MiniNeo Setup ===")
    test_pytorch()
    print("\n")
    test_db_connection()