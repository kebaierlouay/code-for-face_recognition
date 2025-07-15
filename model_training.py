
import os
import face_recognition
import pickle
import cv2
import glob

print("[INFO] Starting face encoding...")

# Get all image paths from dataset folder
imagePaths = glob.glob("dataset/**/*.jpg", recursive=True) + \
             glob.glob("dataset/**/*.png", recursive=True) + \
             glob.glob("dataset/**/*.jpeg", recursive=True)

knownEncodings = []
knownNames = []

for (i, imagePath) in enumerate(imagePaths):
    print(f"[INFO] Processing image {i + 1}/{len(imagePaths)}")

    # Extract name from folder
    name = os.path.basename(os.path.dirname(imagePath))
    
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    boxes = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, boxes)
    
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

print("[INFO] Saving encodings to encodings.pickle...")
data = {"encodings": knownEncodings, "names": knownNames}
with open("encodings.pickle", "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] Done.")
