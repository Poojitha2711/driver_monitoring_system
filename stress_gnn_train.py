import os
import cv2
import torch
import mediapipe as mp
import numpy as np
import random

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# -----------------------------
# PATHS (CHANGE THIS)
# -----------------------------
train_path = "ck_images"
test_path  = "ck_images"

# -----------------------------
# MODEL
# -----------------------------
class StressGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(8, 64)
        self.conv2 = GCNConv(64,128)
        
        self.fc = torch.nn.Linear(128, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        
        x=F.dropout(x,p=0.3,training=self.training)
        
        x = F.relu(self.conv2(x, edge_index))
        x=F.dropout(x,p=0.3,training=self.training)
        #x = F.relu(self.conv3(x, edge_index))
        
        x = torch.mean(x, dim=0)
    
        x = self.fc(x)

        return x

# -----------------------------
# MEDIAPIPE
# -----------------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True)

# -----------------------------
# LABEL FUNCTION
# -----------------------------
def get_stress_label(folder):
    folder=int(folder)
    if folder in [1,3]:
        return 1
    else:
        return 0

# -----------------------------
# GRAPH CREATION FUNCTION
# -----------------------------
def create_graph(img, label):

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    face_landmarks = result.multi_face_landmarks[0]

    landmarks = []
    for lm in face_landmarks.landmark:
        landmarks.append([lm.x - 0.5, lm.y - 0.5])
    h,w,_=img.shape
    features=[]
    for (x_coord,y_coord) in landmarks:
        px=int((x_coord+0.5)*w)
        py=int((y_coord+0.5)*h)
        
        px=max(0,min(px,w-1))
        py=max(0,min(py,h-1))
        
        intensity=img[py,px][0]/255.0
        features.append([x_coord,
                         y_coord,
                         x_coord*y_coord,
                         x_coord**2,
                         y_coord**2,
                         intensity,
                         abs(x_coord),
                         abs(y_coord)
                         ])
    features=np.array(features)
    features=(features-features.mean())/(features.std()+1e-6)
    x = torch.tensor(features, dtype=torch.float)

    edges = []
    for i in range(len(landmarks)):
        for j in range(i+1, min(i+5, len(landmarks))):
            edges.append([i, j])
            edges.append([j, i])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    graph = Data(x=x, edge_index=edge_index)
    graph.y = torch.tensor([label], dtype=torch.float)

    return graph



# -----------------------------
# LOAD TRAIN DATA
# -----------

# -----------------------------
# LOAD TEST DATA
# -----------------------------



import random

# -----------------------------
# TRAIN MODEL
# -----------------------------

def train_model():
    
    train_data = []

    for folder in os.listdir(train_path):

        folder_path = os.path.join(train_path, folder)
        label = get_stress_label(folder)

        for img_name in os.listdir(folder_path):

            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            graph = create_graph(img, label)

            if graph:
                train_data.append(graph)
                
    random.shuffle(train_data)

    split=int(0.8*len(train_data))
    all_data=train_data.copy()
    train_data=all_data[:split]
    test_data=all_data[split:]
            
    test_data = []

    for folder in os.listdir(test_path):

        older_path = os.path.join(test_path, folder)
        label = get_stress_label(folder)

        for img_name in os.listdir(folder_path):

            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            graph = create_graph(img, label)

            if graph:
                test_data.append(graph)

    model = StressGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for epoch in range(50):
        total_loss = 0

        for data in train_data:
            optimizer.zero_grad()

            output = model(data)
            loss = loss_fn(output, data.y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

# -----------------------------
# SAVE MODEL
# -----------------------------
    torch.save(model.state_dict(), "gnn_model_stress_new.pth")
    torch.save(test_data, "stress_test_data.pth")
    
if __name__=="__main__":
    train_model()

# -----------------------------
# TEST ACCURACY
# -----------------------------
'''correct = 0
total = 0

#model.eval()
with torch.no_grad():
    for data in test_data:

        pred = 1 if model(data).item() > 0.5 else 0
        actual = int(data.y.item())

        if pred == actual:
            correct += 1

        total += 1
if total==0:
    print("No test data available")
else:
    accuracy = (correct / total) * 100
    print(f"\nTest Accuracy: {accuracy:.2f}%")'''