import cv2
import mediapipe as mp

# -----------------------------
# Initialize MediaPipe
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

mp_drawing = mp.solutions.drawing_utils

# -----------------------------
# Start Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process image
    result = face_mesh.process(rgb)

    # -----------------------------
    # If face detected
    # -----------------------------
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:

            landmarks = []

            # Extract 468 points
            for lm in face_landmarks.landmark:
                x = int(lm.x * frame.shape[1])   # convert to pixel
                y = int(lm.y * frame.shape[0])
                landmarks.append((x, y))

                # Draw point
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # Print number of landmarks
            print("Total Landmarks:", len(landmarks))  # should be 468

    # -----------------------------
    # Show output
    # -----------------------------
    cv2.imshow("Facial Landmarks", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()