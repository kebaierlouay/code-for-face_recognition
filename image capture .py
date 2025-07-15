import cv2
import os


person_name = "louay"

output_dir = f"dataset/{person_name}"
os.makedirs(output_dir, exist_ok=True)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

count = 0  

print("[INFO] Press 'c' to capture face, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Capture Face - Press 'c' to Save", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and len(faces) > 0:
        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            file_path = os.path.join(output_dir, f"{person_name}_{count}.jpg")
            cv2.imwrite(file_path, face_img)
            print(f"[INFO] Saved: {file_path}")
            count += 1
            break  

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
