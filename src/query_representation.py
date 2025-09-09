import numpy as np
from typing import List, Dict, Tuple, Set

class QueryRepresentation:
    """Class for representing queries in a vector format for the neural network."""

    def __init__(self, db_schema, aliases=None):
        """
        Initialize the query representation with database schema information.

        Args:
            db_schema: A dictionary containing table and column information.
            aliases: A dictionary mapping table aliases to table names
        """
        self.db_schema = db_schema
        self.tables = list(db_schema.keys())
        self.table_to_idx = {table: idx for idx, table in enumerate(self.tables)}
        self.common_aliases = aliases or {}

        # Build a list of all columns across all tables
        self.columns = []
        for table, columns in db_schema.items():
            for column in columns:
                self.columns.append(f"{table}.{column}")

        self.column_to_idx = {column: idx for idx, column in enumerate(self.columns)}

    def extract_query_info(self, query: str) -> Tuple[List[str], List[Tuple[str, str]], List[Tuple[str, str, str]]]:
        """
        Extract tables, join conditions, and predicates from a SQL query.
        """
        tables = []
        join_conditions = []
        predicates = []
        table_aliases = {}

        try:
            # Normalize query
            query = ' '.join(query.lower().split())

            # Extract FROM clause
            from_parts = query.split(' from ')[1]
            # Handle any comments in the query
            if '--' in from_parts:
                from_parts = from_parts.split('--')[0]

            # Get the part up to WHERE, GROUP BY, ORDER BY, etc.
            if ' where ' in from_parts:
                from_parts = from_parts.split(' where ')[0]
            elif ' group by ' in from_parts:
                from_parts = from_parts.split(' group by ')[0]
            elif ' order by ' in from_parts:
                from_parts = from_parts.split(' order by ')[0]

            # Split on commas, handling AS clauses
            table_clauses = [clause.strip() for clause in from_parts.split(',')]

            # Process each table reference
            for clause in table_clauses:
                parts = clause.strip().split(' as ')
                if len(parts) == 2:
                    # Handle "table AS alias"
                    table_name = parts[0].strip()
                    alias = parts[1].strip()
                else:
                    # Handle "table alias" or just "table"
                    parts = clause.strip().split()
                    table_name = parts[0].strip()
                    alias = parts[-1].strip() if len(parts) > 1 else table_name

                # Clean up names
                table_name = table_name.strip('"`[]() ')
                alias = alias.strip('"`[]() ')

                # Map to real table name if it's a common alias
                if table_name in self.common_aliases:
                    table_name = self.common_aliases[table_name]

                # Only add if it's a valid table
                if table_name in self.db_schema:
                    tables.append(table_name)
                    table_aliases[alias] = table_name
                else:
                    print(f"Warning: Unknown table '{table_name}'")

            # Extract predicates and join conditions from WHERE clause
            if ' where ' in query:
                where_clause = query.split(' where ')[1]
                if ' group by ' in where_clause:
                    where_clause = where_clause.split(' group by ')[0]
                elif ' order by ' in where_clause:
                    where_clause = where_clause.split(' order by ')[0]

                # Split conditions, properly handling parentheses
                conditions = []
                current = []
                paren_level = 0

                for word in where_clause.split():
                    if '(' in word:
                        paren_level += word.count('(')
                    if ')' in word:
                        paren_level -= word.count(')')

                    if word.lower() == 'and' and paren_level == 0:
                        if current:
                            conditions.append(' '.join(current))
                            current = []
                    else:
                        current.append(word)

                if current:
                    conditions.append(' '.join(current))

                # Process each condition
                for condition in conditions:
                    # Skip IN and complex LIKE conditions for now
                    if ' in (' in condition or ' like ' in condition:
                        continue

                    # Handle basic equality conditions
                    if '=' in condition:
                        left, right = [s.strip() for s in condition.split('=')]

                        # Check if this is a join condition (table.column = table.column)
                        if '.' in left and '.' in right:
                            left_table, left_col = left.split('.')
                            right_table, right_col = right.split('.')

                            # Map aliases to real table names
                            left_table = table_aliases.get(left_table, left_table)
                            right_table = table_aliases.get(right_table, right_table)

                            if left_table in self.db_schema and right_table in self.db_schema:
                                # Add join condition using real table names
                                join_conditions.append((
                                    f"{left_table}.{left_col}",
                                    f"{right_table}.{right_col}"
                                ))

            # Sort tables to ensure consistent ordering
            tables.sort()

            print(f"Extracted tables: {tables}")
            print(f"Extracted {len(join_conditions)} join conditions")
            print(f"Extracted {len(predicates)} predicates")

            return tables, join_conditions, predicates

        except Exception as e:
            print(f"Error parsing query: {e}")
            # Return empty lists instead of all tables
            return [], [], []

    def encode_join_graph(self, join_conditions: List[Tuple[str, str]]) -> np.ndarray:
        """
        Encode the join graph as an adjacency matrix.
        """
        n_tables = len(self.tables)
        join_graph = np.zeros((n_tables, n_tables), dtype=np.float32)

        # If no join conditions, return empty graph
        if not join_conditions:
            return join_graph

        for cond in join_conditions:
            try:
                left, right = cond

                # Skip if we don't have proper string values
                if not isinstance(left, str) or not isinstance(right, str):
                    continue

                # Extract table names from column references
                table1 = left.split('.')[0] if '.' in left else left
                table2 = right.split('.')[0] if '.' in right else right

                # Clean up table names
                table1 = table1.strip().strip('\'"()` ')
                table2 = table2.strip().strip('\'"()` ')

                # Map aliases to actual table names
                if table1 in self.common_aliases:
                    table1 = self.common_aliases[table1]
                if table2 in self.common_aliases:
                    table2 = self.common_aliases[table2]

                if table1 in self.table_to_idx and table2 in self.table_to_idx:
                    idx1 = self.table_to_idx[table1]
                    idx2 = self.table_to_idx[table2]

                    # Symmetric matrix - set both directions
                    join_graph[idx1, idx2] = 1
                    join_graph[idx2, idx1] = 1
                else:
                    # Only print warning if tables exist in schema but not in table_to_idx
                    if (table1 in self.schema or table2 in self.schema) and \
                    (table1 not in self.table_to_idx or table2 not in self.table_to_idx):
                        print(f"Warning: Table(s) not found in schema: {table1}, {table2}")
            except Exception as e:
                print(f"Error processing join condition {cond}: {e}")

        return join_graph

    def encode_predicates_1hot(self, predicates: List[Tuple[str, str, str]]) -> np.ndarray:
        """
        Encode query predicates using a simple one-hot encoding.

        Args:
            predicates: List of predicate conditions as tuples of (table.column, operator, value)
                        e.g., ("title.production_year", ">", "2000")

        Returns:
            A 1D numpy array with one-hot encoding of predicates.
        """
        n_columns = len(self.columns)
        predicate_vector = np.zeros(n_columns, dtype=np.float32)

        for pred in predicates:
            column = pred[0]
            if column in self.column_to_idx:
                predicate_vector[self.column_to_idx[column]] = 1

        return predicate_vector

    def encode_query(self, join_conditions: List[Tuple[str, str]],
                    predicates: List[Tuple[str, str, str]]) -> np.ndarray:
        """
        Encode a complete query by combining join graph and predicate encodings.

        Args:
            join_conditions: List of join conditions
            predicates: List of predicate conditions

        Returns:
            A 1D numpy array representing the query.
        """
        join_graph = self.encode_join_graph(join_conditions)
        predicate_vector = self.encode_predicates_1hot(predicates)

        # Flatten the upper triangular part of the join graph (since it's symmetric)
        n_tables = len(self.tables)
        join_vector = join_graph[np.triu_indices(n_tables, k=1)]

        # Concatenate the flattened join graph with predicate vector
        query_vector = np.concatenate([join_vector, predicate_vector])

        return query_vector

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