import torch
from torch_geometric.data import Data

def create_graph(landmarks):
    """
    landmarks: list of (x, y) → 468 points from Mediapipe
    """

    # -----------------------------
    # Convert to tensor (nodes)
    # -----------------------------
    x = torch.tensor(landmarks, dtype=torch.float)


    # -----------------------------
    # Create edges (simple chain)
    # -----------------------------
    edges = []

    for i in range(len(landmarks) - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])  # bidirectional

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # -----------------------------
    # Create graph object
    # -----------------------------
    graph = Data(x=x, edge_index=edge_index)

    return graph