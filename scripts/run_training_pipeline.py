import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training_pipeline import TrainingPipeline
from src.config import DATA_DIR, MODELS_DIR

def main():
    """Run the MiniNeo training pipeline."""

    # Set up paths
    query_file = DATA_DIR / "job_queries.sql"
    model_dir = MODELS_DIR

    # Create directories if they don't exist
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    model_dir.mkdir(exist_ok=True, parents=True)

    # Check if query file exists
    if not query_file.exists():
        print(f"Error: Query file {query_file} does not exist.")
        print("Please create this file with SQL queries from the JOB benchmark.")
        return

    # Initialize training pipeline
    pipeline = TrainingPipeline(
        model_dir=model_dir,
        learning_rate=0.001,
        batch_size=16,
        epochs=100
    )

    # Run training loop
    pipeline.training_loop(query_file, num_iterations=20)

if __name__ == "__main__":
    main()