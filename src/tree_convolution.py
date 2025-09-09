import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional

class TreeConvolutionLayer(nn.Module):
    """
    A layer that applies convolution over trees.
    Similar to image convolution but operates on trees.
    """

    def __init__(self, input_dim: int, output_dim: int, num_filters: int):
        """
        Initialize the tree convolution layer.

        Args:
            input_dim: Dimension of input node vectors
            output_dim: Dimension of output node vectors
            num_filters: Number of filters to apply
        """
        super(TreeConvolutionLayer, self).__init__()

        # Each filter consists of weights for parent, left child, and right child
        self.parent_weights = nn.Parameter(torch.randn(num_filters, input_dim))
        self.left_weights = nn.Parameter(torch.randn(num_filters, input_dim))
        self.right_weights = nn.Parameter(torch.randn(num_filters, input_dim))

        # Bias term for each filter
        self.bias = nn.Parameter(torch.zeros(num_filters))

        # Projection to output dimension
        self.projection = nn.Linear(num_filters, output_dim)

        # Initialize weights with Xavier initialization
        nn.init.xavier_uniform_(self.parent_weights)
        nn.init.xavier_uniform_(self.left_weights)
        nn.init.xavier_uniform_(self.right_weights)
        nn.init.xavier_uniform_(self.projection.weight)

    def forward(self, tree_nodes: List[torch.Tensor], tree_structure: List[List[int]]) -> Tuple[List[torch.Tensor], List[List[int]]]:
        """
        Apply tree convolution to a batch of trees.

        Args:
            tree_nodes: List of node feature vectors [batch_size, num_nodes, input_dim]
            tree_structure: List of lists indicating parent-child relationships
                            Each list contains indices of children for the corresponding node

        Returns:
            Tuple of (new_node_features, tree_structure)
        """
        new_nodes = []

        # Process each node in the tree
        for i, node in enumerate(tree_nodes):
            children = tree_structure[i]

            # Get left and right children (if they exist)
            left_child = tree_nodes[children[0]] if len(children) > 0 else torch.zeros_like(node)
            right_child = tree_nodes[children[1]] if len(children) > 1 else torch.zeros_like(node)

            # Apply filters (similar to the example in the Neo paper)
            # Compute the dot product of weights and node vectors
            parent_term = torch.matmul(self.parent_weights, node)
            left_term = torch.matmul(self.left_weights, left_child)
            right_term = torch.matmul(self.right_weights, right_child)

            # Sum the terms and apply activation function
            filter_outputs = F.relu(parent_term + left_term + right_term + self.bias)

            # Project to output dimension
            new_node = self.projection(filter_outputs)
            new_nodes.append(new_node)

        return new_nodes, tree_structure

class DynamicPooling(nn.Module):
    """
    Dynamic pooling layer that flattens a tree into a fixed-size vector.
    """

    def __init__(self, output_dim: int):
        """
        Initialize the dynamic pooling layer.

        Args:
            output_dim: Dimension of the output vector
        """
        super(DynamicPooling, self).__init__()
        self.output_dim = output_dim

    def forward(self, tree_nodes: List[torch.Tensor]) -> torch.Tensor:
        """
        Apply dynamic pooling to a batch of trees.

        Args:
            tree_nodes: List of node feature vectors

        Returns:
            A fixed-size vector representing the tree
        """
        # Stack node features
        if tree_nodes:
            stacked = torch.stack(tree_nodes)

            # Apply max pooling across nodes
            pooled, _ = torch.max(stacked, dim=0)

            return pooled
        else:
            # Return zeros if tree is empty
            return torch.zeros(self.output_dim)

class TreeConvolutionNetwork(nn.Module):
    """
    Neural network that applies tree convolution to query execution plans.
    """

    def __init__(self, query_dim: int, plan_node_dim: int, hidden_dim: int = 128):
        """
        Initialize the tree convolution network.

        Args:
            query_dim: Dimension of query vectors
            plan_node_dim: Dimension of plan node vectors
            hidden_dim: Dimension of hidden layers
        """
        super(TreeConvolutionNetwork, self).__init__()

        # Encoding for query-level features
        self.query_encoder = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.LeakyReLU()
        )

        # Augmented node dimension (original + query features)
        augmented_node_dim = plan_node_dim + hidden_dim // 4

        # Tree convolution layers
        self.tree_conv1 = TreeConvolutionLayer(augmented_node_dim, hidden_dim, 64)
        self.tree_conv2 = TreeConvolutionLayer(hidden_dim, hidden_dim, 64)
        self.tree_conv3 = TreeConvolutionLayer(hidden_dim, hidden_dim, 64)

        # Dynamic pooling
        self.dynamic_pooling = DynamicPooling(hidden_dim)

        # Final layers for cost prediction
        self.final_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, query_vector: torch.Tensor,
                plan_nodes: List[torch.Tensor],
                plan_structure: List[List[int]]) -> torch.Tensor:
        """
        Forward pass through the tree convolution network.

        Args:
            query_vector: Vector representation of the query
            plan_nodes: List of node vectors in the execution plan
            plan_structure: Tree structure of the execution plan

        Returns:
            Predicted cost (latency) of the execution plan
        """
        # Encode query features
        query_features = self.query_encoder(query_vector)

        # Augment each plan node with query features
        augmented_nodes = []
        for node in plan_nodes:
            # Concatenate query features to each node
            augmented_node = torch.cat([node, query_features])
            augmented_nodes.append(augmented_node)

        # Apply tree convolution layers
        conv1_nodes, conv1_structure = self.tree_conv1(augmented_nodes, plan_structure)
        conv2_nodes, conv2_structure = self.tree_conv2(conv1_nodes, conv1_structure)
        conv3_nodes, conv3_structure = self.tree_conv3(conv2_nodes, conv2_structure)

        # Apply dynamic pooling to get a fixed-size representation
        pooled = self.dynamic_pooling(conv3_nodes)

        # Final prediction
        cost = self.final_layers(pooled)

        return cost

def prepare_batch(query_vectors: List[torch.Tensor],
                 plan_nodes_list: List[List[torch.Tensor]],
                 plan_structures: List[List[List[int]]]) -> Tuple[torch.Tensor, List[List[torch.Tensor]], List[List[List[int]]]]:
    """
    Prepare a batch for training the tree convolution network.

    Args:
        query_vectors: List of query vectors
        plan_nodes_list: List of lists of plan node vectors
        plan_structures: List of tree structures

    Returns:
        Batched tensors
    """
    # Stack query vectors
    batched_query = torch.stack(query_vectors)

    # No need to modify plan_nodes_list and plan_structures
    # as they will be processed node-by-node in the model

    return batched_query, plan_nodes_list, plan_structures