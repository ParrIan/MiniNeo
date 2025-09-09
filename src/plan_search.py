# src/plan_search.py
import time
import heapq
import torch
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from copy import deepcopy

from src.query_representation import QueryRepresentation
from src.plan_representation import PlanNode, PlanRepresentation
from src.tree_convolution import TreeConvolutionNetwork

class PlanSearcher:
    """
    A class that implements best-first search for query plans using a trained value network.
    """

    def __init__(self,
                value_network: TreeConvolutionNetwork,
                query_rep: QueryRepresentation,
                plan_rep: PlanRepresentation,
                time_cutoff_ms: int = 250):
        """
        Initialize the plan searcher.

        Args:
            value_network: The trained tree convolution network
            query_rep: Query representation
            plan_rep: Plan representation
            time_cutoff_ms: Time cutoff for search in milliseconds
        """
        self.value_network = value_network
        self.query_rep = query_rep
        self.plan_rep = plan_rep
        self.time_cutoff_ms = time_cutoff_ms

    def search(self, query_vector: np.ndarray, tables: List[str]) -> PlanNode:
        """
        Search for the best query execution plan given a query.

        Args:
            query_vector: Vector representation of the query
            tables: List of tables in the query

        Returns:
            The best found plan
        """
        # Convert query vector to tensor
        query_tensor = torch.tensor(query_vector, dtype=torch.float32)

        # Create initial plan with unspecified scans for each table
        initial_nodes = []
        for table in tables:
            node = PlanNode('scan', [table], 'unspecified')
            initial_nodes.append(node)

        # Initialize priority queue for best-first search
        # Each item in the queue is a tuple of (predicted_cost, plan_index, plan_nodes)
        queue = []

        # Encode the initial plan
        initial_plans = []
        for node in initial_nodes:
            initial_plans.append([node])

        # Add each initial plan to the queue
        for i, plan in enumerate(initial_plans):
            # Encode plan
            encoded_plan = self.encode_plan(plan)

            # Get predicted cost
            with torch.no_grad():
                cost = self.value_network(
                    query_tensor,
                    encoded_plan[0],  # node vectors
                    encoded_plan[1]   # structure
                )

            # Add to queue (negative cost because heapq is a min-heap)
            heapq.heappush(queue, (cost.item(), i, id(plan), plan))

        # Start search timer
        start_time = time.time()

        # Best complete plan found so far
        best_complete_plan = None
        best_complete_cost = float('inf')

        # Number of plans explored
        plans_explored = 0

        # Main search loop
        while queue and time.time() - start_time < self.time_cutoff_ms / 1000:
            # Get plan with lowest predicted cost
            pred_cost, _, _, plan = heapq.heappop(queue)
            plans_explored += 1

            # Check if plan is complete
            if len(plan) == 1 and self.is_complete_plan(plan[0], tables):
                # We found a complete plan with better cost
                if pred_cost < best_complete_cost:
                    best_complete_cost = pred_cost
                    best_complete_plan = plan[0]
                continue

            # Generate child plans
            child_plans = self.generate_children(plan, tables)

            # Evaluate each child plan
            for i, child_plan in enumerate(child_plans):
                # Encode the child plan
                encoded_plan = self.encode_plan(child_plan)

                # Get predicted cost
                with torch.no_grad():
                    cost = self.value_network(
                        query_tensor,
                        encoded_plan[0],  # node vectors
                        encoded_plan[1]   # structure
                    )

                # Check if this is a complete plan
                if len(child_plan) == 1 and self.is_complete_plan(child_plan[0], tables):
                    if cost.item() < best_complete_cost:
                        best_complete_cost = cost.item()
                        best_complete_plan = child_plan[0]
                else:
                    # Add to queue
                    heapq.heappush(queue, (cost.item(), plans_explored + i, id(child_plan), child_plan))

        # If time ran out and we haven't found a complete plan, take the best partial plan
        # and complete it greedily
        if best_complete_plan is None:
            if queue:
                _, _, best_partial_plan = heapq.heappop(queue)
                best_complete_plan = self.complete_plan_greedily(best_partial_plan, tables)
            else:
                # Fallback: create a left-deep plan with hash joins
                best_complete_plan = self.create_fallback_plan(tables)

        # print(f"Search completed. Explored {plans_explored} plans in {time.time() - start_time:.2f} seconds.")
        return best_complete_plan

    def encode_plan(self, plan_nodes: List[PlanNode]) -> Tuple[List[torch.Tensor], List[List[int]]]:
        """
        Encode a plan for the value network.

        Args:
            plan_nodes: List of plan nodes

        Returns:
            Tuple of (node_vectors, structure)
        """
        # Collect all nodes in a flat list
        all_nodes = []
        structure = []

        # Helper function to build the structure
        def traverse(node, parent_idx=-1):
            node_idx = len(all_nodes)
            all_nodes.append(node)
            children_indices = []

            if node.left:
                left_idx = traverse(node.left, node_idx)
                children_indices.append(left_idx)

            if node.right:
                right_idx = traverse(node.right, node_idx)
                children_indices.append(right_idx)

            structure.append(children_indices)
            return node_idx

        # Process each tree in the forest
        roots = []
        for node in plan_nodes:
            root_idx = traverse(node)
            roots.append(root_idx)

        # Encode each node
        node_vectors = []
        for node in all_nodes:
            encoded_node = self.plan_rep.encode_node(node)
            node_vectors.append(torch.tensor(encoded_node, dtype=torch.float32))

        return node_vectors, structure

    def generate_children(self, plan: List[PlanNode], tables: List[str]) -> List[List[PlanNode]]:
        """
        Generate child plans by applying valid operators.

        Args:
            plan: Current plan (forest of trees)
            tables: List of tables in the query

        Returns:
            List of child plans
        """
        children = []

        # If there's only one node and it's complete, no children
        if len(plan) == 1 and self.is_complete_plan(plan[0], tables):
            return children

        # Generate plans by joining pairs of trees
        for i in range(len(plan)):
            for j in range(i + 1, len(plan)):
                # Skip if both nodes are unspecified scans
                if plan[i].is_scan() and plan[j].is_scan() and \
                   plan[i].scan_type == 'unspecified' and plan[j].scan_type == 'unspecified':
                    continue

                # Create plans with different join types
                for join_type in ['hash_join', 'merge_join', 'nested_loop_join']:
                    # Create a joined plan with i as left child and j as right child
                    new_plan = deepcopy(plan)
                    i_node = new_plan[i]
                    j_node = new_plan[j]

                    # Create joined tables list
                    joined_tables = list(set(i_node.tables + j_node.tables))

                    # Create join node
                    join_node = PlanNode(join_type, joined_tables)
                    join_node.left = i_node
                    join_node.right = j_node

                    # Remove the two joined nodes and add the join node
                    new_plan = [node for k, node in enumerate(new_plan) if k != i and k != j]
                    new_plan.append(join_node)

                    children.append(new_plan)

        # Generate plans by specifying scan types for unspecified scans
        for i, node in enumerate(plan):
            if node.is_scan() and node.scan_type == 'unspecified':
                # Create plan with table scan
                table_scan_plan = deepcopy(plan)
                table_scan_plan[i].scan_type = 'table_scan'
                children.append(table_scan_plan)

                # Create plan with index scan
                index_scan_plan = deepcopy(plan)
                index_scan_plan[i].scan_type = 'index_scan'
                children.append(index_scan_plan)

        return children

    def is_complete_plan(self, node: PlanNode, tables: List[str]) -> bool:
        """
        Check if a plan is complete.

        Args:
            node: Root node of the plan
            tables: List of tables in the query

        Returns:
            True if the plan is complete, False otherwise
        """
        # Check if all tables are included
        plan_tables = set(node.tables)
        if len(plan_tables) != len(tables):
            return False

        for table in tables:
            if table not in plan_tables:
                return False

        # Check if all scan types are specified
        def check_scans(node):
            if node.is_scan():
                return node.scan_type != 'unspecified'
            else:
                return check_scans(node.left) and check_scans(node.right)

        return check_scans(node)

    def complete_plan_greedily(self, plan: List[PlanNode], tables: List[str]) -> PlanNode:
        """
        Complete a partial plan greedily when time runs out.

        Args:
            plan: Partial plan
            tables: List of tables in the query

        Returns:
            A complete plan
        """
        # First, specify all unspecified scans as table scans
        for node in plan:
            def specify_scans(n):
                if n.is_scan() and n.scan_type == 'unspecified':
                    n.scan_type = 'table_scan'
                elif not n.is_scan():
                    if n.left:
                        specify_scans(n.left)
                    if n.right:
                        specify_scans(n.right)

            specify_scans(node)

        # If there's only one node left, it should be a complete plan
        if len(plan) == 1:
            return plan[0]

        # Otherwise, join the remaining trees with hash joins in a left-deep fashion
        result = plan[0]
        for i in range(1, len(plan)):
            joined_tables = list(set(result.tables + plan[i].tables))
            join_node = PlanNode('hash_join', joined_tables)
            join_node.left = result
            join_node.right = plan[i]
            result = join_node

        return result

    def create_fallback_plan(self, tables: List[str]) -> PlanNode:
        """
        Create a fallback plan if no plan was found during search.

        Args:
            tables: List of tables in the query

        Returns:
            A complete plan
        """
        # Create scan nodes
        scan_nodes = []
        for table in tables:
            scan_node = PlanNode('scan', [table], 'table_scan')
            scan_nodes.append(scan_node)

        # Create a left-deep plan with hash joins
        result = scan_nodes[0]
        for i in range(1, len(scan_nodes)):
            joined_tables = list(set(result.tables + scan_nodes[i].tables))
            join_node = PlanNode('hash_join', joined_tables)
            join_node.left = result
            join_node.right = scan_nodes[i]
            result = join_node

        return result