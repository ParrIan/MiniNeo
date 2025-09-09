import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import psycopg2
import random
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from src.config import MODELS_DIR, DB_CONFIG
from src.query_representation import QueryRepresentation, get_imdb_schema
from src.plan_representation import PlanRepresentation, PlanNode
from src.tree_convolution import TreeConvolutionNetwork, prepare_batch
from src.plan_search import PlanSearcher
from src.db_utils import get_connection, execute_query

class TrainingPipeline:
    """
    Training pipeline for the MiniNeo optimizer.
    Handles collecting execution data and training the value network.
    """

    def __init__(self,
                  model_dir: Optional[Path] = None,
                  learning_rate: float = 0.001,
                  batch_size: int = 16,
                  epochs: int = 50):
      """
      Initialize the training pipeline.
      """
      self.model_dir = model_dir or MODELS_DIR
      self.model_dir.mkdir(exist_ok=True, parents=True)

      self.learning_rate = learning_rate
      self.batch_size = batch_size
      self.epochs = epochs

      # Initialize schema and representations
      self.schema = get_imdb_schema()
      # Print schema tables to verify
      print("Loaded schema with tables:", sorted(list(self.schema.keys())))

      self.common_aliases = {
        "lt": "link_type",
        "ml": "movie_link",
        "t": "title",
        "mk": "movie_keyword",
        "k": "keyword",
        "mc": "movie_companies",
        "cn": "company_name",
        "ct": "company_type",
        "mi": "movie_info",
        "miidx": "movie_info_idx",
        "it": "info_type",
        "n": "name",
        "pi": "person_info",
        "ci": "cast_info",
        "an": "aka_name",
        "rt": "role_type",
        "cct": "comp_cast_type",
        "cc": "complete_cast",
        "kt": "kind_type"
    }

      self.query_rep = QueryRepresentation(self.schema, self.common_aliases)
      self.plan_rep = PlanRepresentation(self.schema)

      # Initialize experience collection
      self.experience = []

      # Initialize network dimensions
      self.query_dim = None
      self.plan_node_dim = None

      # Initialize network, optimizer, and loss function
      self.value_network = None
      self.optimizer = None
      self.loss_fn = nn.MSELoss()

    def collect_initial_experience(self, query_file: Path):
        """
        Collect initial experience from PostgreSQL plans.

        Args:
            query_file: Path to file containing SQL queries
        """
        print("Collecting initial experience from PostgreSQL plans...")

        # Load queries
        with open(query_file, 'r') as f:
            queries = f.read().strip().split(';')
            queries = [q.strip() for q in queries if q.strip()]

        # Connect to database
        conn = get_connection()
        cursor = conn.cursor()

        # For each query, get PostgreSQL's plan and execute it
        for i, query in enumerate(tqdm(queries)):
            # Get PostgreSQL's execution plan
            explain_query = f"EXPLAIN (FORMAT JSON) {query}"
            cursor.execute(explain_query)
            pg_plan_json = cursor.fetchone()[0]

            # Parse the plan to extract join order, scan methods, etc.
            plan_node = self.parse_postgres_plan(pg_plan_json[0]['Plan'])

            # Extract tables and conditions from the query
            tables, join_conditions, predicates = self.extract_query_info(query)

            # Encode the query
            query_vector = self.query_rep.encode_query(join_conditions, predicates)

            # Measure execution time
            start_time = time.time()
            cursor.execute(query)
            _ = cursor.fetchall()
            execution_time = time.time() - start_time

            # Store experience
            self.experience.append({
                'query_id': i,
                'query': query,
                'query_vector': query_vector,
                'tables': tables,
                'join_conditions': join_conditions,
                'predicates': predicates,
                'plan_node': plan_node,
                'latency': execution_time
            })

            # Initialize dimensions if not already done
            if self.query_dim is None:
                self.query_dim = len(query_vector)

            if self.plan_node_dim is None:
                test_node = PlanNode('scan', ['title'], 'table_scan')
                self.plan_node_dim = len(self.plan_rep.encode_node(test_node))

        cursor.close()
        conn.close()

        print(f"Collected experience from {len(self.experience)} queries.")

        # Initialize network
        self.initialize_network()

    def initialize_network(self):
        """Initialize the tree convolution network and optimizer."""
        print("Initializing value network...")

        self.value_network = TreeConvolutionNetwork(self.query_dim, self.plan_node_dim)
        self.optimizer = optim.Adam(self.value_network.parameters(), lr=self.learning_rate)

        print("Value network initialized.")

    def train_network(self, is_initial_training: bool = True):
        """
        Train the value network using collected experience.

        Args:
            is_initial_training: Whether this is the initial training or retraining
        """
        print(f"{'Initial training' if is_initial_training else 'Retraining'} value network...")

        # If retraining, reset network weights
        if not is_initial_training:
            self.initialize_network()

        # Prepare training data
        train_data = []
        for exp in self.experience:
            query_vector = torch.tensor(exp['query_vector'], dtype=torch.float32)

            # Encode the plan
            encoded_plan = self.encode_plan_for_training(exp['plan_node'])

            # Get latency
            latency = torch.tensor([exp['latency']], dtype=torch.float32)

            train_data.append((query_vector, encoded_plan, latency))

        # Train for multiple epochs
        best_loss = float('inf')
        best_state = None

        for epoch in range(self.epochs):
            # Shuffle training data
            random.shuffle(train_data)

            # Training batches
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, len(train_data), self.batch_size):
                batch = train_data[i:i+self.batch_size]

                # Zero gradients
                self.optimizer.zero_grad()

                # Accumulate loss across batch
                batch_loss = 0.0

                for query_vector, (plan_nodes, plan_structure), latency in batch:
                    # Forward pass
                    pred_latency = self.value_network(query_vector, plan_nodes, plan_structure)

                    # Compute loss
                    loss = self.loss_fn(pred_latency, latency)
                    batch_loss += loss

                # Average loss across batch
                batch_loss /= len(batch)

                # Backward pass and optimize
                batch_loss.backward()
                self.optimizer.step()

                epoch_loss += batch_loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_epoch_loss:.6f}")

            # Save best model
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_state = self.value_network.state_dict()

        # Restore best model
        self.value_network.load_state_dict(best_state)

        # Save model
        model_path = self.model_dir / "value_network.pt"
        torch.save({
            'model_state_dict': self.value_network.state_dict(),
            'query_dim': self.query_dim,
            'plan_node_dim': self.plan_node_dim,
        }, model_path)

        print(f"Training completed. Best loss: {best_loss:.6f}")
        print(f"Model saved to {model_path}")

    def optimize_queries(self, query_file: Path, output_file: Optional[Path] = None):
        """
        Optimize queries using the trained value network.

        Args:
            query_file: Path to file containing SQL queries
            output_file: Path to save optimization results
        """
        print("Optimizing queries with MiniNeo...")

        # Load queries
        with open(query_file, 'r') as f:
            queries = f.read().strip().split(';')
            queries = [q.strip() for q in queries if q.strip()]

        # Initialize plan searcher
        plan_searcher = PlanSearcher(self.value_network, self.query_rep, self.plan_rep)

        # Connect to database
        conn = get_connection()
        cursor = conn.cursor()

        # Results to save
        results = []

        # For each query, optimize with MiniNeo and compare with PostgreSQL
        for i, query in enumerate(tqdm(queries)):
            # Extract tables and conditions from the query
            tables, join_conditions, predicates = self.extract_query_info(query)

            # Encode the query
            query_vector = self.query_rep.encode_query(join_conditions, predicates)

            # Search for the best plan
            best_plan = plan_searcher.search(query_vector, tables)

            # Measure execution time of PostgreSQL's plan
            start_time = time.time()
            cursor.execute(query)
            _ = cursor.fetchall()
            pg_time = time.time() - start_time

            # Convert MiniNeo plan to PostgreSQL hints
            hints = self.plan_to_pg_hints(best_plan)

            # Measure execution time of MiniNeo's plan
            hint_query = f"/*+ {hints} */ {query}"
            start_time = time.time()
            cursor.execute(hint_query)
            _ = cursor.fetchall()
            neo_time = time.time() - start_time

            # Store results
            results.append({
                'query_id': i,
                'query': query,
                'pg_time': pg_time,
                'neo_time': neo_time,
                'speedup': pg_time / neo_time if neo_time > 0 else 0,
                'neo_plan': self.plan_to_string(best_plan)
            })

            # Add to experience
            self.experience.append({
                'query_id': len(self.experience),
                'query': query,
                'query_vector': query_vector,
                'tables': tables,
                'join_conditions': join_conditions,
                'predicates': predicates,
                'plan_node': best_plan,
                'latency': neo_time
            })

        cursor.close()
        conn.close()

        # Print summary statistics
        speedups = [r['speedup'] for r in results]
        avg_speedup = sum(speedups) / len(speedups)
        geo_mean_speedup = np.exp(np.mean(np.log([max(s, 0.001) for s in speedups])))

        print(f"Optimization completed for {len(queries)} queries.")
        print(f"Average speedup: {avg_speedup:.2f}x")
        print(f"Geometric mean speedup: {geo_mean_speedup:.2f}x")

        # Save results if output file provided
        if output_file:
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")

        return results

    def training_loop(self, query_file: Path, num_iterations: int = 10):
        """
        Run the complete training loop with iterative improvement.

        Args:
            query_file: Path to file containing SQL queries
            num_iterations: Number of training iterations
        """
        print(f"Starting MiniNeo training loop for {num_iterations} iterations...")

        # Collect initial experience
        self.collect_initial_experience(query_file)

        # Initial training
        self.train_network(is_initial_training=True)

        # Iterative optimization and retraining
        for i in range(num_iterations):
            print(f"\nIteration {i+1}/{num_iterations}")

            # Optimize queries
            output_file = self.model_dir / f"results_iter_{i+1}.csv"
            results = self.optimize_queries(query_file, output_file)

            # Retrain network
            self.train_network(is_initial_training=False)

        print("Training loop completed.")

    def parse_postgres_plan(self, plan_json: Dict) -> PlanNode:
        """
        Parse a PostgreSQL execution plan in JSON format.

        Args:
            plan_json: PostgreSQL plan in JSON format

        Returns:
            A PlanNode representation of the plan
        """
        node_type = plan_json['Node Type']

        if node_type in ['Seq Scan', 'Index Scan', 'Index Only Scan', 'Bitmap Heap Scan']:
            # Handle scan node
            table = plan_json.get('Relation Name', '')
            scan_type = 'table_scan' if node_type == 'Seq Scan' else 'index_scan'
            return PlanNode('scan', [table], scan_type)

        elif node_type in ['Hash Join', 'Merge Join', 'Nested Loop']:
            # Handle join node
            join_type_map = {
                'Hash Join': 'hash_join',
                'Merge Join': 'merge_join',
                'Nested Loop': 'nested_loop_join'
            }
            join_type = join_type_map[node_type]

            # Parse children
            left_child = self.parse_postgres_plan(plan_json['Plans'][0])
            right_child = self.parse_postgres_plan(plan_json['Plans'][1])

            # Combine table lists
            tables = list(set(left_child.tables + right_child.tables))

            # Create join node
            join_node = PlanNode(join_type, tables)
            join_node.left = left_child
            join_node.right = right_child

            return join_node

        else:
            # Handle other node types as simple pass-through
            tables = []
            children = []

            if 'Relation Name' in plan_json:
                tables.append(plan_json['Relation Name'])

            if 'Plans' in plan_json:
                for child_plan in plan_json['Plans']:
                    child_node = self.parse_postgres_plan(child_plan)
                    children.append(child_node)
                    tables.extend(child_node.tables)

            # Create a generic node (treated as a join node in our model)
            node = PlanNode('hash_join', list(set(tables)))

            if len(children) >= 1:
                node.left = children[0]
            if len(children) >= 2:
                node.right = children[1]

            return node

    def extract_query_info(self, query: str) -> Tuple[List[str], List[Tuple[str, str]], List[Tuple[str, str, str]]]:
        """
        Extract tables, join conditions, and predicates from a SQL query.
        This is a simplified parser and may need enhancement for complex queries.
        """
        try:
            # print(f"Processing query: {query[:50]}...")

            tables = []
            table_aliases = {}
            join_conditions = []
            predicates = []

            # Extract tables from FROM clause
            if 'FROM' in query.upper():
                from_parts = query.upper().split('FROM')
                from_clause = from_parts[1]

                # Extract part until WHERE, JOIN, GROUP BY, etc.
                if 'WHERE' in from_clause:
                    from_clause = from_clause.split('WHERE')[0]
                elif 'GROUP BY' in from_clause:
                    from_clause = from_clause.split('GROUP BY')[0]
                elif 'ORDER BY' in from_clause:
                    from_clause = from_clause.split('ORDER BY')[0]
                elif 'JOIN' in from_clause:
                    from_clause = from_clause.split('JOIN')[0]

                # Handle complex FROM clauses with nested queries
                # Simple approach: extract words that look like table names
                # This is a heuristic and won't work for all cases
                for potential_table in self.schema.keys():
                    if potential_table.upper() in from_clause:
                        tables.append(potential_table)

                # Also try to extract aliases
                parts = from_clause.split(',')
                for part in parts:
                    words = part.strip().split()
                    if len(words) >= 2:
                        potential_table = words[0].lower()
                        potential_alias = words[-1].lower()

                        # Clean up potential table and alias names
                        potential_table = potential_table.strip('()"\'`')
                        potential_alias = potential_alias.strip('()"\'`')

                        if potential_table in self.schema:
                            tables.append(potential_table)
                            table_aliases[potential_alias] = potential_table

            # Default approach: if we couldn't extract tables, include all schema tables
            if not tables:
                print("Could not extract tables from query, using all schema tables")
                tables = list(self.schema.keys())

            # Extract join conditions and predicates - simplified approach
            # For now, let's just return the tables and minimal join conditions
            # to get the training pipeline working

            return tables, join_conditions, predicates
        except Exception as e:
            print(f"Error parsing query: {e}")
            # Return all tables as fallback
            return list(self.schema.keys()), [], []

    def encode_plan_for_training(self, plan_node: PlanNode) -> Tuple[List[torch.Tensor], List[List[int]]]:
        """
        Encode a plan for training.

        Args:
            plan_node: Root node of the plan

        Returns:
            Tuple of (node_vectors, structure)
        """
        nodes = []
        structure = []

        # Helper function to traverse the tree
        def traverse(node, parent_idx=-1):
            node_idx = len(nodes)
            nodes.append(node)
            children_indices = []

            if node.left:
                left_idx = traverse(node.left, node_idx)
                children_indices.append(left_idx)

            if node.right:
                right_idx = traverse(node.right, node_idx)
                children_indices.append(right_idx)

            structure.append(children_indices)
            return node_idx

        # Traverse the tree
        traverse(plan_node)

        # Encode each node
        node_vectors = []
        for node in nodes:
            encoded_node = self.plan_rep.encode_node(node)
            node_vectors.append(torch.tensor(encoded_node, dtype=torch.float32))

        return node_vectors, structure

    def plan_to_pg_hints(self, plan_node: PlanNode) -> str:
        """
        Convert a plan to PostgreSQL hints.

        Args:
            plan_node: Root node of the plan

        Returns:
            String of PostgreSQL hints
        """
        # This is a simplified version - actual hint generation would be more complex
        hints = []

        # Helper function to traverse the tree
        def traverse(node):
            if node.is_join():
                # Add join order hint
                if node.left and node.right:
                    left_tables = ','.join(node.left.tables)
                    right_tables = ','.join(node.right.tables)
                    hints.append(f"Leading({left_tables} {right_tables})")

                # Add join method hint
                if node.node_type == 'hash_join':
                    hints.append(f"HashJoin({','.join(node.tables)})")
                elif node.node_type == 'merge_join':
                    hints.append(f"MergeJoin({','.join(node.tables)})")
                elif node.node_type == 'nested_loop_join':
                    hints.append(f"NestedLoop({','.join(node.tables)})")

                # Traverse children
                if node.left:
                    traverse(node.left)
                if node.right:
                    traverse(node.right)

            elif node.is_scan():
                # Add scan method hint
                if node.scan_type == 'table_scan':
                    hints.append(f"SeqScan({','.join(node.tables)})")
                elif node.scan_type == 'index_scan':
                    hints.append(f"IndexScan({','.join(node.tables)})")

        # Traverse the plan
        traverse(plan_node)

        return ' '.join(hints)

    def plan_to_string(self, plan_node: PlanNode) -> str:
        """
        Convert a plan to a readable string.

        Args:
            plan_node: Root node of the plan

        Returns:
            String representation of the plan
        """
        result = []

        # Helper function to traverse the tree
        def traverse(node, level=0):
            indent = "  " * level
            result.append(f"{indent}{node}")

            if node.left:
                traverse(node.left, level + 1)
            if node.right:
                traverse(node.right, level + 1)

        # Traverse the plan
        traverse(plan_node)

        return '\n'.join(result)