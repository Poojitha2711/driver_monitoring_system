import cv2
import torch
import numpy as np
import mediapipe as mp
import winsound
import pyttsx3
import threading
import time

from torch_geometric.data import Data
from drowsy_gnn_train import DrowsyGNN
from stress_gnn_train import StressGNN
from graph_creation import create_graph

# -----------------------------
# Load Models
# -----------------------------
drowsy_model = DrowsyGNN()
drowsy_model.load_state_dict(torch.load("gnn_model_drowsy_new.pth", map_location="cpu"))
drowsy_model.eval()

stress_model = StressGNN()
stress_model.load_state_dict(torch.load("gnn_model_stress_new.pth", map_location="cpu"))
stress_model.eval()

# -----------------------------
# Voice System
# -----------------------------
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.setProperty('rate', 120)

speech_lock = threading.Lock()

def speak(text):
    with speech_lock:
        engine.stop()
        engine.say(text)
        engine.runAndWait()

# -----------------------------
# Alarm System
# -----------------------------
alarm_on = False
alarm_event = threading.Event()

def alarm():
    while not alarm_event.is_set():
        winsound.Beep(2000, 100)
        time.sleep(0.05)

# -----------------------------
# MediaPipe
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not working")
    exit()

print("Camera opened")

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Create Features for GNN
        features_2 = []
        features_8 = []

        for lm in landmarks:
            x = lm.x - 0.5
            y = lm.y - 0.5

            features_2.append([x, y])
            features_8.append([x, y, x*y, x**2, y**2, 0.0, abs(x), abs(y)])

        graph_drowsy = create_graph(features_2)
        graph_stress = create_graph(features_8)

        # -----------------------------
        # GNN Predictions
        # -----------------------------
        with torch.no_grad():
            d_out = drowsy_model(graph_drowsy).item()
            s_out = stress_model(graph_stress).item()

        # -----------------------------
        # Convert to Labels
        # -----------------------------
        drowsiness = "DROWSY" if d_out > 0.5 else "ALERT"

        if s_out < 0.4:
            stress = "LOW"
        elif s_out < 0.7:
            stress = "MEDIUM"
        else:
            stress = "HIGH"

        # -----------------------------
        # Driver State
        # -----------------------------
        if drowsiness == "DROWSY":
            driver_state = "CRITICAL"
        elif stress == "HIGH":
            driver_state = "WARNING"
        else:
            driver_state = "SAFE"

        # -----------------------------
        # Alert System
        # -----------------------------
        if driver_state == "CRITICAL":
            if not alarm_on:
                alarm_on = True
                alarm_event.clear()

                threading.Thread(target=alarm, daemon=True).start()

                threading.Thread(
                    target=speak,
                    args=("Driver is drowsy. Please wake up immediately!",),
                    daemon=True
                ).start()

        elif driver_state == "WARNING":
            alarm_on = False

            threading.Thread(
                target=speak,
                args=("Driver stress level is high. Please stay calm.",),
                daemon=True
            ).start()

        else:
            alarm_on = False
            alarm_event.set()

        # -----------------------------
        # Display
        # -----------------------------
        cv2.putText(frame, f"Drowsiness: {drowsiness}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

        cv2.putText(frame, f"Stress: {stress}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (128, 128, 128), 2)

        cv2.putText(frame, f"Driver State: {driver_state}",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)

    cv2.imshow("Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()