import numpy as np
from typing import List, Dict, Tuple, Union, Optional

class PlanNode:
    """A node in a query execution plan tree."""

    def __init__(self, node_type: str, tables: List[str] = None, scan_type: str = None):
        """
        Initialize a plan node.

        Args:
            node_type: Type of the node (join operator or scan)
            tables: List of tables involved in this node
            scan_type: Type of scan (for leaf nodes only)
        """
        self.node_type = node_type
        self.tables = tables or []
        self.scan_type = scan_type
        self.left = None  # Left child
        self.right = None  # Right child

    def is_join(self):
        """Check if this node is a join operator."""
        return self.node_type in ['hash_join', 'merge_join', 'nested_loop_join']

    def is_scan(self):
        """Check if this node is a scan operator."""
        return self.node_type == 'scan'

    def __repr__(self):
        if self.is_scan():
            return f"{self.scan_type}({','.join(self.tables)})"
        else:
            return f"{self.node_type}({','.join(self.tables)})"

class PlanRepresentation:
    """Class for representing query execution plans in a format suitable for the neural network."""

    def __init__(self, db_schema):
        """
        Initialize the plan representation with database schema information.

        Args:
            db_schema: A dictionary containing table information.
        """
        self.db_schema = db_schema
        self.tables = list(db_schema.keys())
        self.table_to_idx = {table: idx for idx, table in enumerate(self.tables)}

        # Define join types
        self.join_types = ['hash_join', 'merge_join', 'nested_loop_join']
        self.join_type_to_idx = {join_type: idx for idx, join_type in enumerate(self.join_types)}

        # Define scan types
        self.scan_types = ['table_scan', 'index_scan', 'unspecified']
        self.scan_type_to_idx = {scan_type: idx for idx, scan_type in enumerate(self.scan_types)}

    def encode_node(self, node: PlanNode) -> np.ndarray:
        """
        Encode a plan node as a vector.

        Args:
            node: A PlanNode to encode

        Returns:
            A vector representation of the node.
        """
        # Vector size: |join_types| + 2*|tables|
        vector_size = len(self.join_types) + 2 * len(self.tables)
        node_vector = np.zeros(vector_size, dtype=np.float32)

        if node.is_join():
            # Set the join type
            join_idx = self.join_type_to_idx.get(node.node_type, 0)
            node_vector[join_idx] = 1

            # Set the tables information - union of child tables
            for table in node.tables:
                if table in self.table_to_idx:
                    table_idx = self.table_to_idx[table]
                    # Both scan types are set to 1 for join nodes
                    node_vector[len(self.join_types) + table_idx] = 1  # Table scan position
                    node_vector[len(self.join_types) + len(self.tables) + table_idx] = 1  # Index scan position

        elif node.is_scan():
            # For scan nodes, set the appropriate scan type
            table = node.tables[0] if node.tables else ""
            if table in self.table_to_idx:
                table_idx = self.table_to_idx[table]

                if node.scan_type == 'table_scan':
                    node_vector[len(self.join_types) + table_idx] = 1
                elif node.scan_type == 'index_scan':
                    node_vector[len(self.join_types) + len(self.tables) + table_idx] = 1
                elif node.scan_type == 'unspecified':
                    # Both scan types are set to 1 for unspecified
                    node_vector[len(self.join_types) + table_idx] = 1
                    node_vector[len(self.join_types) + len(self.tables) + table_idx] = 1

        return node_vector

    def encode_plan_tree(self, root: PlanNode) -> List[Tuple[np.ndarray, List[int]]]:
        """
        Encode a plan tree as a list of node vectors with parent-child relationships.

        Args:
            root: The root node of the plan tree

        Returns:
            A list of tuples (node_vector, [children_indices])
        """
        # BFS traversal to encode the tree
        queue = [root]
        encoded_nodes = []
        node_to_idx = {}

        # First pass: encode all nodes
        for node in queue:
            node_vector = self.encode_node(node)
            node_idx = len(encoded_nodes)
            node_to_idx[node] = node_idx
            encoded_nodes.append((node_vector, []))

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        # Second pass: add child indices
        for node in queue:
            node_idx = node_to_idx[node]
            if node.left:
                left_idx = node_to_idx[node.left]
                encoded_nodes[node_idx][1].append(left_idx)
            if node.right:
                right_idx = node_to_idx[node.right]
                encoded_nodes[node_idx][1].append(right_idx)

        return encoded_nodes

    def build_example_plan(self) -> PlanNode:
        """
        Build an example plan for testing.

        Returns:
            A sample plan tree.
        """
        # Create leaf nodes (scans)
        title_scan = PlanNode('scan', ['title'], 'index_scan')
        movie_companies_scan = PlanNode('scan', ['movie_companies'], 'table_scan')
        company_name_scan = PlanNode('scan', ['company_name'], 'table_scan')

        # Create join nodes
        join1 = PlanNode('hash_join', ['title', 'movie_companies'])
        join1.left = title_scan
        join1.right = movie_companies_scan

        join2 = PlanNode('nested_loop_join', ['title', 'movie_companies', 'company_name'])
        join2.left = join1
        join2.right = company_name_scan

        return join2

def get_imdb_schema():
    """
    Returns a schema for the IMDB dataset based on the actual database tables.
    """
    return {
        "title": ["id", "title", "imdb_index", "kind_id", "production_year", "imdb_id", "phonetic_code",
                 "episode_of_id", "season_nr", "episode_nr", "series_years", "md5sum"],
        "movie_companies": ["id", "movie_id", "company_id", "company_type_id", "note"],
        "movie_info": ["id", "movie_id", "info_type_id", "info", "note"],
        "movie_info_idx": ["id", "movie_id", "info_type_id", "info", "note"],
        "movie_keyword": ["id", "movie_id", "keyword_id"],
        "cast_info": ["id", "person_id", "movie_id", "person_role_id", "note", "nr_order", "role_id"],
        "company_name": ["id", "name", "country_code", "imdb_id", "name_pcode_nf", "name_pcode_sf", "md5sum"],
        "name": ["id", "name", "imdb_index", "imdb_id", "gender", "name_pcode_cf", "name_pcode_nf", "surname_pcode", "md5sum"],
        "keyword": ["id", "keyword", "phonetic_code"],
        "info_type": ["id", "info"],
        "company_type": ["id", "kind"],
        "kind_type": ["id", "kind"],
        "person_info": ["id", "person_id", "info_type_id", "info", "note"],
        "aka_name": ["id", "person_id", "name", "imdb_index", "name_pcode_cf", "name_pcode_nf", "surname_pcode", "md5sum"],
        "aka_title": ["id", "movie_id", "title", "imdb_index", "kind_id", "production_year", "phonetic_code", "season_nr", "episode_nr", "note", "md5sum"],
        "char_name": ["id", "name", "imdb_index", "imdb_id", "name_pcode_nf", "surname_pcode", "md5sum"],
        "comp_cast_type": ["id", "kind"],
        "complete_cast": ["id", "movie_id", "subject_id", "status_id"],
        "link_type": ["id", "link"],
        "movie_link": ["id", "movie_id", "linked_movie_id", "link_type_id"],
        "role_type": ["id", "role"]
    }
