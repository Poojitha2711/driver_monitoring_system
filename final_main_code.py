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
# Load GNN Models
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
engine.setProperty('rate',120)

speech_lock = threading.Lock()

def speak(text):
    with speech_lock:
        engine.stop()
        engine.say(text)
        engine.runAndWait()

# -----------------------------
# Alarm
# -----------------------------
alarm_on = False
alarm_event=threading.Event()

def alarm():
    global alarm_on
    while not alarm_event.is_set():
        winsound.Beep(2000,100)
        time.sleep(0.05)

# -----------------------------
# Mediapipe Setup
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)
# -----------------------------
# Create Graph Edges
# -----------------------------
edge_list = []

for i in range(467):
    edge_list.append([i,i+1])
    edge_list.append([i+1,i])

edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
# -----------------------------
# Prediction Buffers
# -----------------------------
drowsy_buffer = []
stress_buffer = []

BUFFER_SIZE = 25
frame_count=0
prev_coords=None
stress_history=[]
baseline_eyebrow=None
prev_stress="LOW"
stress_buffer=[]
stress="LOW"
# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not working")
    exit()
else:
    print("Camera opened")
    
while True:

    ret, frame = cap.read()
    if not ret:
        print("Frame not received")
        continue
    
    frame_count+=1
    if frame_count%3!=0:
        continue

    if not ret:
        break

    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks=[]
        for lm in results.multi_face_landmarks[0].landmark:
            x=lm.x
            y=lm.y
            landmarks.append([x,y])
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        features_2=[]
        features_8=[]
        for lm in face_landmarks:
            x=lm.x-0.5
            y=lm.y-0.5
            
            features_2.append([x,y])
            features_8.append([x,y,x*y,x**2,y**2,0.0,abs(x),abs(y)])

        graph_drowsy=create_graph(features_2)
        graph_stress=create_graph(features_8)
            
        with torch.no_grad():
            d_out=drowsy_model(graph_drowsy).item()
            s_out=stress_model(graph_stress).item()
        coords_np = np.array([[lm.x, lm.y] for lm in face_landmarks])
        coords_np=coords_np-coords_np.mean(axis=0)
        coords_np=coords_np/ (np.std(coords_np)+ 1e-6)
        nose_x=coords_np[1][0]
        left_x=coords_np[234][0]
        right_x=coords_np[454][0]

        # 🔴 SAFETY FIX (VERY IMPORTANT)
        if coords_np.shape[0] < 468:
            continue   # skip this frame completely
        coords_np=coords_np[:468]
        # smoothing
        if prev_coords is not None:
            coords_np = coords_np * 0.8 + prev_coords * 0.2

        prev_coords = coords_np

        coords_np = torch.tensor(coords_np, dtype=torch.float)
        # ----------------------------
        # Drowsiness Prediction
        # ----------------------------
        def eye_aspect_ratio(eye):
            A = np.linalg.norm(eye[1] - eye[5])
            B = np.linalg.norm(eye[2] - eye[4])
            C = np.linalg.norm(eye[0] - eye[3])
            return (A + B) / (2.0 * C)
        coords_np=coords_np.numpy()
        left_eye = coords_np[[33,160,158,133,153,144]]
        right_eye = coords_np[[362,385,387,263,373,380]]

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2
        
        def mouth_open_ratio(mouth):
            A = np.linalg.norm(mouth[13] - mouth[14])
            B = np.linalg.norm(mouth[78] - mouth[308])
            return A / B

        mouth = coords_np
        mar = mouth_open_ratio(mouth)
        
        # eyebrow distance (tension)
        left_eyebrow = coords_np[65]
        right_eyebrow = coords_np[295]
        eyebrow_dist = np.linalg.norm(left_eyebrow - right_eyebrow)
        # mouth width (emotion)
        mouth_left = coords_np[78]
        mouth_right = coords_np[308]
        mouth_width = np.linalg.norm(mouth_left - mouth_right)
        
        face_width = np.linalg.norm(coords_np[234] - coords_np[454])

        eyebrow_ratio = eyebrow_dist / face_width
        mouth_ratio = mouth_width / face_width

        coords=torch.tensor(features_2,dtype=torch.float)
        coords = coords - coords.mean(dim=0)
        coords = coords / (coords.std(dim=0) + 1e-6)
        if torch.mean(torch.abs(coords))<0.01:
            continue

        data = Data(x=coords, edge_index=edge_index)

        data.batch = torch.zeros(coords.shape[0], dtype=torch.long)

        with torch.no_grad():
            d_out = drowsy_model(data)

        
        drowsy_buffer.append(ear)
        if len(drowsy_buffer) > BUFFER_SIZE:
            drowsy_buffer.pop(0)

        # average confidence
        avg_ear = sum(drowsy_buffer) / len(drowsy_buffer)

        # stability check
        stable_count = sum(1 for x in drowsy_buffer if x > 0.7)

        #final decision
        if avg_ear<0.23:
            drowsiness = "DROWSY"
        else:
            drowsiness = "ALERT"

        # -----------------------------
        # Stress Prediction
        # -----------------------------
        with torch.no_grad():
            s_out = stress_model(graph_stress).item()

        s_score=s_out
       
        # -------------------------
        # STRESS 
        # -------------------------

        # FACE WIDTH (reference for normalization)
        face_width = np.linalg.norm(coords_np[234] - coords_np[454])

        # FEATURES (normalized
        eyebrow = abs(coords_np[70][1] - coords_np[159][1])/face_width
        eye=abs(coords_np[159][1]-coords_np[145][1])/face_width
        nose=abs(coords_np[1][1]-coords_np[5][1])/face_width
        top_lip=coords_np[13]
        bottom_lip=coords_np[14]
        lip=np.linalg.norm(top_lip-bottom_lip)
        lip=lip/face_width
        
        current_stress=eyebrow+nose+lip
    
    # -------------------------
    # DIFFERENCE
    # -------------------------
        if baseline_eyebrow is None:
            baseline_eyebrow=eyebrow
        eyebrow_dev=abs(eyebrow-baseline_eyebrow)
        
        stress_score = (4.0 * eyebrow +
                        3.0*eye+
                        2.0*nose+
                        0.7*lip
                        )
       
        if stress_score < 1.2:
            stress = "LOW"
        elif stress_score < 1.35:
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
                threading.Thread(
                    target=alarm,
                    daemon=True
                ).start()

                threading.Thread(
                    target=speak,
                    args=("Driver is drowsy. Please wake up immediately.",),
                    daemon=True
                ).start()

        elif driver_state == "WARNING":
            alarm_on=False
                
            threading.Thread(
                target=speak,
                args=("Driver stress level is high. Please stay calm.",),
                daemon=True
            ).start()

        else:
            alarm_on = False
            alarm_event.set()
            
        # -----------------------------
        # Display Output
        # -----------------------------
        cv2.putText(frame,f"Drowsiness: {drowsiness}",
                    (20,40),cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,(0,255,0),2)

        cv2.putText(frame,f"Stress: {stress}",
                    (20,80),cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,(128,0,128),2)

        cv2.putText(frame,f"Driver State: {driver_state}",
                    (20,120),cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,(0,0,255),2)

    cv2.imshow("Driver Monitoring System",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()