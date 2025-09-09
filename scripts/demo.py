"""
MiniNeo Demo Script - Used for presentation demonstration.
Shows the comparison between PostgreSQL and MiniNeo on a selected JOB query.
"""
import sys
import os
import time
import torch
import pandas as pd
import traceback
import argparse
import random
from pathlib import Path

from src.config import MODELS_DIR, DB_CONFIG
from src.query_representation import QueryRepresentation, get_imdb_schema
from src.plan_representation import PlanRepresentation, PlanNode
from src.tree_convolution import TreeConvolutionNetwork
from src.plan_search import PlanSearcher
from src.db_utils import get_connection
from src.training_pipeline import TrainingPipeline

def print_header(text):
    """Print a formatted header for better readability."""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80)

def print_plan_tree(plan, level=0):
    """Print a plan tree in a readable format."""
    indent = "  " * level

    if isinstance(plan, dict):
        # PostgreSQL plan
        node_type = plan.get('Node Type', '')
        relation = plan.get('Relation Name', '')
        est_rows = plan.get('Plan Rows', '')
        cost = f"{plan.get('Startup Cost', 0):.1f}..{plan.get('Total Cost', 0):.1f}"

        node_str = f"{indent}[{node_type}]"
        if relation:
            node_str += f" on {relation}"
        node_str += f" (rows: {est_rows}, cost: {cost})"
        print(node_str)

        if 'Plans' in plan:
            for child_plan in plan['Plans']:
                print_plan_tree(child_plan, level + 1)
    else:
        # MiniNeo plan
        if hasattr(plan, 'is_join') and plan.is_join():
            node_str = f"{indent}[{plan.node_type}] on {', '.join(plan.tables)}"
            print(node_str)

            if plan.left:
                print_plan_tree(plan.left, level + 1)
            if plan.right:
                print_plan_tree(plan.right, level + 1)
        elif hasattr(plan, 'is_scan') and plan.is_scan():
            node_str = f"{indent}[{plan.scan_type}] on {', '.join(plan.tables)}"
            print(node_str)
        else:
            print(f"{indent}[Unknown node type]")

def load_model(model_path):
    """Load a trained MiniNeo model."""
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path)

    model = TreeConvolutionNetwork(
        checkpoint['query_dim'],
        checkpoint['plan_node_dim']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model

def visualize_join_order_comparison(pg_plan, neo_plan):
    """Create a visualization comparing join orders."""
    def extract_pg_join_order(plan, tables=None):
        if tables is None:
            tables = []

        if 'Relation Name' in plan:
            tables.append(plan['Relation Name'])

        if 'Plans' in plan:
            for child in plan['Plans']:
                extract_pg_join_order(child, tables)

        return tables

    def extract_neo_join_order(plan, join_order=None, tables=None):
        if join_order is None:
            join_order = []
        if tables is None:
            tables = set()

        if plan.is_scan():
            for table in plan.tables:
                if table not in tables:
                    tables.add(table)
                    join_order.append(table)
            return join_order

        if plan.left:
            extract_neo_join_order(plan.left, join_order, tables)
        if plan.right:
            extract_neo_join_order(plan.right, join_order, tables)

        return join_order

    pg_join_order = extract_pg_join_order(pg_plan)
    neo_join_order = extract_neo_join_order(neo_plan)

    print("\nJoin Order Comparison:")
    print("-" * 40)
    print("PostgreSQL order:".ljust(20), " → ".join(pg_join_order))
    print("MiniNeo order:".ljust(20), " → ".join(neo_join_order))
    print("-" * 40)

def load_queries():
    """Load JOB queries from file."""
    project_root = Path(__file__).parent.parent
    query_file = project_root / "data" / "job_queries.sql"

    with open(query_file, 'r') as f:
        queries = []
        current_query = []
        current_id = None

        for line in f:
            line = line.strip()

            # Look for query ID comments
            if line.startswith('--') and 'Query' in line:
                if current_query:
                    queries.append((current_id, '\n'.join(current_query)))
                    current_query = []
                current_id = line.split('Query')[-1].strip()
            elif line:
                current_query.append(line)

        # Add the last query
        if current_query:
            queries.append((current_id, '\n'.join(current_query)))

    return queries

def run_demo(query_num=None, num_queries=1, show_plans=True):
    """Main demo function."""
    try:
        print_header("MiniNeo Demonstration")
        print("Setting up environment...")

        schema = get_imdb_schema()
        query_rep = QueryRepresentation(schema)
        plan_rep = PlanRepresentation(schema)

        # Load best model
        model_path = None
        model_dirs = [
            "15_iterations_1_75_epochs",
            "15_iterations_1_100_epochs",
            "15_iterations_2_100_epochs",
            "15_iterations_3_100_epochs",
            "15_iterations_4_100_epochs",
            "."
        ]

        for dir_name in model_dirs:
            candidate_path = MODELS_DIR / dir_name / "value_network.pt"
            if candidate_path.exists():
                model_path = candidate_path
                break

        if model_path is None:
            model_path = MODELS_DIR / "value_network.pt"
            if not model_path.exists():
                raise FileNotFoundError(f"Could not find a model file in {MODELS_DIR}")

        model = load_model(model_path)

        # Connect to database
        print("Connecting to database...")
        conn = get_connection()
        cursor = conn.cursor()

        # Load queries
        queries = load_queries()

        if query_num:
            # Find specific query
            selected_queries = [(qid, q) for qid, q in queries if qid.lower() == query_num.lower()]
            if not selected_queries:
                print(f"Query {query_num} not found. Available queries: {', '.join(qid for qid, _ in queries)}")
                return False
        else:
            # Random selection
            selected_queries = random.sample(queries, min(num_queries, len(queries)))

        # Process each selected query
        for qid, demo_query in selected_queries:
            print_header(f"Query {qid}")
            print(demo_query)

            print_header("Query Analysis")
            tables, join_conditions, predicates = query_rep.extract_query_info(demo_query)
            query_vector = query_rep.encode_query(join_conditions, predicates)

            print(f"Tables involved: {', '.join(tables)}")
            print(f"Number of join conditions: {len(join_conditions)}")
            print(f"Number of predicates: {len(predicates)}")

            if show_plans:
                print_header("PostgreSQL Plan")
                explain_query = f"EXPLAIN (FORMAT JSON) {demo_query}"
                cursor.execute(explain_query)
                pg_plan_json = cursor.fetchone()[0]

                print("PostgreSQL's plan:")
                print_plan_tree(pg_plan_json[0]['Plan'])

            print_header("PostgreSQL Execution")
            print("Executing query with PostgreSQL's plan...")
            start_time = time.time()
            cursor.execute(demo_query)
            results = cursor.fetchall()
            pg_time = time.time() - start_time

            print(f"Retrieved {len(results)} results")
            print(f"PostgreSQL execution time: {pg_time:.3f} seconds")

            if show_plans:
                print_header("MiniNeo Plan Generation")
                plan_searcher = PlanSearcher(model, query_rep, plan_rep)

                print("Searching for optimal plan with MiniNeo...")
                search_start = time.time()
                best_plan = plan_searcher.search(query_vector, tables)
                search_time = time.time() - search_start

                print(f"Plan search completed in {search_time:.3f} seconds")
                print("\nMiniNeo's plan:")
                print_plan_tree(best_plan)

            print_header("MiniNeo Execution")
            pipeline = TrainingPipeline()
            hints = pipeline.plan_to_pg_hints(best_plan)
            hint_query = f"/*+ {hints} */ {demo_query}"

            print("Executing query with MiniNeo's plan...")
            print(f"Query with hints: {hint_query[:100]}...")

            start_time = time.time()
            cursor.execute(hint_query)
            results = cursor.fetchall()
            neo_time = time.time() - start_time

            print(f"Retrieved {len(results)} results")
            print(f"MiniNeo execution time: {neo_time:.3f} seconds")

            print_header("Performance Comparison")
            speedup = pg_time / neo_time if neo_time > 0 else float('inf')

            print(f"PostgreSQL time: {pg_time:.3f} seconds")
            print(f"MiniNeo time:    {neo_time:.3f} seconds")
            print(f"Speedup:         {speedup:.2f}x")

            if show_plans:
                print_header("Plan Analysis")
                visualize_join_order_comparison(pg_plan_json[0]['Plan'], best_plan)

        cursor.close()
        conn.close()

        print_header("Demo Complete")
        print("Thank you for watching the MiniNeo demonstration!")

        return True

    except Exception as e:
        print(f"Error during demo: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MiniNeo demonstration')
    parser.add_argument('--query-num', type=str, help='Specific query number to run (e.g., 1a, 19b)')
    parser.add_argument('--num-queries', type=int, default=1, help='Number of random queries to run')
    parser.add_argument('--show-plans', type=bool, default=True, help='Whether to show query plans')

    args = parser.parse_args()
    run_demo(args.query_num, args.num_queries, args.show_plans)