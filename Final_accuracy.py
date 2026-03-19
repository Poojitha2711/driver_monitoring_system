import torch
from drowsy_gnn_train import DrowsyGNN
from stress_gnn_train import StressGNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD MODELS
# -----------------------------
drowsy_model = DrowsyGNN().to(device)
drowsy_model.load_state_dict(torch.load("gnn_model_drowsy_new.pth"))
drowsy_model.eval()

stress_model = StressGNN().to(device)
stress_model.load_state_dict(torch.load("gnn_model_stress_new.pth"))
stress_model.eval()

drowsy_test_data=torch.load("drowsy_test_data.pth",weights_only=False)
stress_test_data=torch.load("stress_test_data.pth",weights_only=False)

# -----------------------------
# DROWSY ACCURACY
# -----------------------------
def evaluate(model, data_list):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in data_list:
            data = data.to(device)

            output = model(data)
            pred = (output > 0.5).float()

            if pred.item() == data.y.item():
                correct += 1

            total += 1
    
    return (correct / total) * 100

# -----------------------------
# LOAD TEST DATA
# -----------------------------

drowsy_acc = evaluate(drowsy_model, drowsy_test_data)
stress_acc = evaluate(stress_model, stress_test_data)

# -----------------------------
# FINAL OVERALL ACCURACY
# -----------------------------
total_correct = 0
total_samples = 0

# Drowsy contribution
total_correct += (drowsy_acc / 100) * len(drowsy_test_data)
total_samples += len(drowsy_test_data)

# Stress contribution
total_correct += (stress_acc / 100) * len(stress_test_data)
total_samples += len(stress_test_data)

overall_acc = (total_correct / total_samples) * 100


# -----------------------------
# PRINT RESULTS
# -----------------------------
print("\n===== FINAL RESULTS =====")
print(f"Drowsiness Accuracy : {drowsy_acc:.2f}%")
print(f"Stress Accuracy     : {stress_acc:.2f}%")
print(f"Overall Accuracy    : {overall_acc:.2f}%")