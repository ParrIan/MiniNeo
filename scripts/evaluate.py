import sys
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODELS_DIR
from src.query_representation import QueryRepresentation, get_imdb_schema
from src.plan_representation import PlanRepresentation
from src.tree_convolution import TreeConvolutionNetwork
from src.training_pipeline import TrainingPipeline

def evaluate_performance():
    """Evaluate MiniNeo performance across training iterations."""

    # Set up paths
    model_dir = MODELS_DIR / "15_iterations_1_100_epochs"
    results_dir = model_dir

    # Load results from each iteration
    results = []

    for i in range(1, 16):
        result_file = results_dir / f"results_iter_{i}.csv"
        if result_file.exists():
            df = pd.read_csv(result_file)
            df['iteration'] = i
            results.append(df)

    if not results:
        print("No results files found. Run training first.")
        return

    # Combine results
    all_results = pd.concat(results)

    # Calculate performance metrics
    metrics = []

    for i, group in all_results.groupby('iteration'):
        speedups = group['speedup'].values
        geo_mean = np.exp(np.mean(np.log([max(s, 0.001) for s in speedups])))

        metrics.append({
            'iteration': i,
            'mean_speedup': group['speedup'].mean(),
            'geo_mean_speedup': geo_mean,
            'min_speedup': group['speedup'].min(),
            'max_speedup': group['speedup'].max(),
            'queries_improved': (group['speedup'] > 1).sum(),
            'total_queries': len(group)
        })

    metrics_df = pd.DataFrame(metrics)

    # Print summary
    print("MiniNeo Performance Evaluation")
    print("=============================")
    print(f"Evaluated {len(metrics)} training iterations")
    print(f"Final geometric mean speedup: {metrics_df.iloc[-1]['geo_mean_speedup']:.2f}x")
    print(f"Final arithmetic mean speedup: {metrics_df.iloc[-1]['mean_speedup']:.2f}x")
    print(f"Queries improved: {metrics_df.iloc[-1]['queries_improved']} / {metrics_df.iloc[-1]['total_queries']} ({100 * metrics_df.iloc[-1]['queries_improved'] / metrics_df.iloc[-1]['total_queries']:.1f}%)")

    # Plot performance over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['iteration'], metrics_df['geo_mean_speedup'], 'o-', label='Geometric Mean Speedup')
    plt.plot(metrics_df['iteration'], metrics_df['mean_speedup'], 's-', label='Arithmetic Mean Speedup')
    plt.axhline(y=1.0, color='r', linestyle='--', label='PostgreSQL Baseline')
    plt.xlabel('Training Iteration')
    plt.ylabel('Speedup (X times faster)')
    plt.title('MiniNeo Performance Over Training Iterations')
    plt.legend()
    plt.grid(True)

    # Save plot
    plot_file = results_dir / "performance_plot.png"
    plt.savefig(plot_file)
    print(f"Performance plot saved to {plot_file}")

    # Save metrics
    metrics_file = results_dir / "performance_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Performance metrics saved to {metrics_file}")

if __name__ == "__main__":
    evaluate_performance()