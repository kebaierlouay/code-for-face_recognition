# recognize_faces_webcam.py

import face_recognition
import cv2
import numpy as np
import pickle
import time

# Load known encodings
print("[INFO] Loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Start webcam
cap = cv2.VideoCapture(0)

cv_scaler = 4  # Resize factor to improve speed
face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0

def process_frame(frame):
    global face_locations, face_encodings, face_names
    
    small_frame = cv2.resize(frame, (0, 0), fx=1/cv_scaler, fy=1/cv_scaler)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
    
    face_names = []
    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, encoding)
        if face_distances.size > 0:
            best_match = np.argmin(face_distances)
            if matches[best_match]:
                name = known_face_names[best_match]

        face_names.append(name)
    return frame

def draw_results(frame):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.rectangle(frame, (left, top - 35), (right, top), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, top - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
    return frame

def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed > 1:
        fps = frame_count / elapsed
        frame_count = 0
        start_time = time.time()
    return fps

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed = process_frame(frame)
    frame = draw_results(processed)
    
    current_fps = calculate_fps()
    cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Webcam Face Recognition", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
