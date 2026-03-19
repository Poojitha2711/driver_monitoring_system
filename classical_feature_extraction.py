import cv2
import mediapipe as mp
import numpy as np

# -----------------------------
# Initialize MediaPipe
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# -----------------------------
# Blink variables
# -----------------------------
blink_count = 0
eye_closed = False
EAR_THRESHOLD = 0.2

# -----------------------------
# Distance function
# -----------------------------
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# -----------------------------
# EAR
# -----------------------------
def compute_EAR(landmarks):
    p1 = landmarks[33]
    p2 = landmarks[160]
    p3 = landmarks[158]
    p4 = landmarks[133]
    p5 = landmarks[153]
    p6 = landmarks[144]

    A = distance(p2, p6)
    B = distance(p3, p5)
    C = distance(p1, p4)

    return (A + B) / (2.0 * C + 1e-6)

# -----------------------------
# MAR
# -----------------------------
def compute_MAR(landmarks):
    p1 = landmarks[13]
    p2 = landmarks[14]
    p3 = landmarks[78]
    p4 = landmarks[308]

    A = distance(p1, p2)
    C = distance(p3, p4)

    return A / (C + 1e-6)

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:

            landmarks = []

            for lm in face_landmarks.landmark:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                landmarks.append((x, y))

            # -----------------------------
            # Compute EAR & MAR
            # -----------------------------
            ear = compute_EAR(landmarks)
            mar = compute_MAR(landmarks)

            # -----------------------------
            # Blink Detection
            # -----------------------------
            if ear < EAR_THRESHOLD:
                if not eye_closed:
                    eye_closed = True
            else:
                if eye_closed:
                    blink_count += 1
                    eye_closed = False

            # -----------------------------
            # Display Output
            # -----------------------------
            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.putText(frame, f"Blinks: {blink_count}", (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Driver Monitoring - Classical Features", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()