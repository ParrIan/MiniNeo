import sys
import os
import requests
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR

def download_job_queries():
    """Download JOB queries from GitHub."""

    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(exist_ok=True, parents=True)

    # URL for JOB queries
    base_url = "https://raw.githubusercontent.com/gregrahn/join-order-benchmark/master/"

    # Output file
    output_file = DATA_DIR / "job_queries.sql"

    # Download queries
    print(f"Downloading JOB queries to {output_file}...")

    with open(output_file, 'w') as f:
        # JOB has queries numbered 1a, 1b, ..., 33c
        for i in range(1, 34):
            for suffix in ['a', 'b', 'c', 'd']:
                query_url = f"{base_url}{i}{suffix}.sql"
                try:
                    response = requests.get(query_url)
                    if response.status_code == 200:
                        query = response.text.strip()
                        f.write(f"-- Query {i}{suffix}\n")
                        f.write(f"{query};\n\n")
                        print(f"Downloaded query {i}{suffix}")
                    else:
                        # Skip if query doesn't exist
                        pass
                except Exception as e:
                    print(f"Error downloading query {i}{suffix}: {e}")

    print(f"JOB queries downloaded to {output_file}")

if __name__ == "__main__":
    download_job_queries()