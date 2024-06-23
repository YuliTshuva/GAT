"""
Yuli Tshuva
Reformatting downloaded data
"""

import torch
from os.path import join
from torch_geometric.datasets import Planetoid
import numpy as np

# Define the path to save the datasets
DATA_DIR = join("..", "data")
DATASETS = ["Citeseer", "Pubmed"]

for dataset_name in DATASETS:
    # Load the dataset
    dataset = Planetoid(root=DATA_DIR, name=dataset_name)

    # Get the data object
    data = dataset[0]

    # Access node features and labels
    node_features = data.x.numpy()  # Convert to numpy array for easier manipulation
    node_labels = data.y.numpy()  # Convert to numpy array for easier manipulation

    # Access masks
    train_mask = data.train_mask.numpy()
    val_mask = data.val_mask.numpy()
    test_mask = data.test_mask.numpy()

    # Combine node features and labels into a single array
    nodes_data = np.hstack((np.arange(node_features.shape[0]).reshape(-1, 1),  # Add node IDs
                            node_features,  # Add node features
                            node_labels.reshape(-1, 1)))  # Add node labels

    # Save masks
    np.save(join(DATA_DIR, dataset_name, 'train_mask.npy'), train_mask)
    np.save(join(DATA_DIR, dataset_name, 'val_mask.npy'), val_mask)
    np.save(join(DATA_DIR, dataset_name, 'test_mask.npy'), test_mask)

    # Access edges
    edges = data.edge_index.numpy().T  # Transpose to get each edge as a row

    dataset_path = join(DATA_DIR, dataset_name)

    # Write node features and labels to a file
    with open(join(dataset_path, f'{dataset_name}.content'), 'w') as f:
        for node in nodes_data:
            f.write('\t'.join(map(str, node)) + '\n')

    # Write edges to a file
    with open(join(dataset_path, f'{dataset_name}.cites'), 'w') as f:
        for edge in edges:
            f.write('\t'.join(map(str, edge)) + '\n')
