"""
Yuli Tshuva
In the GAT implementation, the paper uses the Cora, Citeseer, and Pubmed datasets.
Unfortunately, only the Cora dataset is on PyGat implementation.
Then, let's install Citeseer and Pubmed datasets.
Notice that the default values of the datasets loading is the same as the paper.
There was a mistake in the paper in the number of edges of both datasets.
"""

from os.path import join
from torch_geometric.datasets import Planetoid

# Define the path to save the datasets
DATA_DIR = join("..", "data")
DATASETS = ["Citeseer", "Pubmed"]

# Get the datasets (iteratively)
for dataset_name in DATASETS:
    # Load the dataset
    dataset = Planetoid(root=DATA_DIR,
                        name=dataset_name)

    # Get the data object
    data = dataset[0]

    print(f'Dataset: {dataset}')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Training nodes: {data.train_mask.sum()}')
    print(f'Validation nodes: {data.val_mask.sum()}')
    print(f'Test nodes: {data.test_mask.sum()}')
