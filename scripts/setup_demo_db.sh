cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

echo "Setting up IMDB database for MiniNeo demo..."

# Check if the imdb database already exists
if psql -lqt | cut -d \| -f 1 | grep -qw imdb; then
    echo "Database 'imdb' already exists. Skipping creation."
else
    echo "Creating 'imdb' database..."
    createdb imdb

    # Check if schema.sql exists and apply it
    if [ -f "$PROJECT_ROOT/data/imdb/schema.sql" ]; then
        echo "Applying schema..."
        psql -d imdb -f "$PROJECT_ROOT/data/imdb/schema.sql"
    fi

    # Check if IMDB.dump exists and restore it
    if [ -f "$PROJECT_ROOT/data/imdb/IMDB.dump" ]; then
        echo "Restoring database from dump file..."
        pg_restore -d imdb "$PROJECT_ROOT/data/imdb/IMDB.dump"
    else
        echo "Error: IMDB.dump file not found at $PROJECT_ROOT/data/imdb/IMDB.dump"
        exit 1
    fi

    echo "Database setup complete!"
fi

# Verify database was set up correctly
echo "Verifying database setup..."
if psql -d imdb -c "SELECT count(*) FROM title;" > /dev/null 2>&1; then
    echo "Database verification successful."
else
    echo "Error: Database verification failed. Please check if the database was restored correctly."
    exit 1
fi

echo "IMDB database is ready for the demo."