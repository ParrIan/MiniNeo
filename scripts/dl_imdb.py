import os
import subprocess
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR, DB_CONFIG
from src.db_utils import execute_query, get_connection

def download_imdb_dataset():
    """Download and extract the IMDB dataset from Harvard Dataverse."""
    imdb_dir = DATA_DIR / "imdb"
    imdb_dir.mkdir(exist_ok=True, parents=True)

    os.chdir(imdb_dir)

    dataset_file = imdb_dir / "IMDB.dump"
    if not dataset_file.exists():
        print("Downloading IMDB dataset from Harvard Dataverse...")
        harvard_url = "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/2QYZBT/TGYUNU"
        subprocess.run(["curl", "-L", "-o", "IMDB.dump", harvard_url], check=True)
        print("Download complete!")

    # Download schema file if it doesn't exist
    if not (imdb_dir / "schema.sql").exists():
        print("Downloading schema file...")
        subprocess.run(["curl", "-OL", "https://raw.githubusercontent.com/gregrahn/join-order-benchmark/master/schema.sql"], check=True)

    print("IMDB dataset download complete!")
    return imdb_dir

def restore_imdb_database():
    """Restore the IMDB database from the downloaded dump file."""
    imdb_dir = DATA_DIR / "imdb"
    dump_file = imdb_dir / "IMDB.dump"

    print("Restoring IMDB database from dump file...")

    db_params = f"-h {DB_CONFIG['host']} -p {DB_CONFIG['port']} -U {DB_CONFIG['user']} -d {DB_CONFIG['database']}"

    try:
        # Execute pg_restore to restore the database
        subprocess.run(f"pg_restore -v {db_params} {dump_file}", shell=True, check=True)
        print("Database restored successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error restoring database: {e}")
        print("You might need to create tables manually using the schema.sql file")
        return False

if __name__ == "__main__":
    imdb_dir = download_imdb_dataset()
    restore_imdb_database()