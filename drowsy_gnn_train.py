import os
import cv2
import torch
import mediapipe as mp
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# -----------------------------
# GNN MODEL
# -----------------------------
class DrowsyGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(2, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64,64)
        self.fc = torch.nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        x = torch.mean(x, dim=0)  # global pooling
        x = self.fc(x)

        return torch.sigmoid(x)

# -----------------------------
# GRAPH CREATION
# -----------------------------
def create_graph(landmarks):
    x = torch.tensor(landmarks, dtype=torch.float)

    edges = []
    for i in range(len(landmarks)):
        for j in range(i+1,min(i+5,len(landmarks))):
            edges.append([i, j])
            edges.append([j, i])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)

# -----------------------------
# INITIALIZE MEDIAPIPE
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# -----------------------------
# LOAD DATASET
# -----------------------------
dataset_path = r"C:\Users\pooji\Desktop\Final Year Project\Multi class\train"   # your folder

data_list = []

for label_name in ["drowsy", "notdrowsy"]:

    label = 1 if label_name == "drowsy" else 0
    folder_path = os.path.join(dataset_path, label_name)

    for file in os.listdir(folder_path):

        img_path = os.path.join(folder_path, file)
        

        img = cv2.imread(img_path)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            continue

        # Extract landmarks
        landmarks = []
        for lm in result.multi_face_landmarks[0].landmark:
            landmarks.append([lm.x-0.5, lm.y-0.5])

        # Convert to graph
        graph = create_graph(landmarks)

        # Assign label
        graph.y = torch.tensor([label], dtype=torch.float)

        data_list.append(graph)

#print("Total samples:", len(data_list))
import random
random.shuffle(data_list)
split=int(0.8*len(data_list))
train_data=data_list[:split]
test_data=data_list[split:]
# -----------------------------
# TRAIN MODEL
# -----------------------------
def train_model():
    model = DrowsyGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCELoss()
    for epoch in range(60):
        total_loss = 0

        for data in train_data:
            optimizer.zero_grad()

            output = model(data)
            loss = loss_fn(output, data.y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss=total_loss/len(train_data)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

# -----------------------------
# SAVE MODEL
# -----------------------------
    torch.save(model.state_dict(), "gnn_model_drowsy_new.pth")
    torch.save(test_data,"drowsy_test_data.pth")
if __name__=="__main__":
    train_model()
