import face_recognition
import os
import pickle
import cv2

faces_dir = "faces"
model_file = "trained_faces.pkl"

all_encodings = []
all_names = []

if os.path.exists(model_file):
    with open(model_file, "rb") as f:
        all_encodings, all_names = pickle.load(f)
        print(f"[INFO] Loaded {len(all_names)} existing faces")

for person_name in os.listdir(faces_dir):
    person_path = os.path.join(faces_dir, person_name)
    if not os.path.isdir(person_path):
        continue

    print(f"[ENCODING] {person_name}")
    for img_file in os.listdir(person_path):
        img_path = os.path.join(person_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[SKIP] Cannot read {img_path}")
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)  # Faster than cnn
        if not boxes:
            print(f"[SKIP] No face found in {img_file}")
            continue

        encodings = face_recognition.face_encodings(rgb, boxes)
        all_encodings.extend(encodings)
        all_names.extend([person_name] * len(encodings))

print(f"[INFO] Encoding complete. Total faces: {len(all_names)}")

with open(model_file, "wb") as f:
    pickle.dump((all_encodings, all_names), f)

print(f"[DONE] Saved encodings to {model_file}")
print(f"[INFO] Total faces: {len(all_names)}")
print(f"[INFO] Encoding complete. Total faces: {len(all_names)}")  