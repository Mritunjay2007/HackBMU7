import cv2
import face_recognition
import speech_recognition as sr
import pyttsx3
import pickle
import numpy as np
import os

engine = pyttsx3.init()

def speak(text):
    print(f"[Parrot]: {text}")
    engine.say(text)
    engine.runAndWait()

def select_microphone(preferred_name=None):
    mics = sr.Microphone.list_microphone_names()
    for i, name in enumerate(mics):
        print(f"[{i}] {name}")
        if preferred_name and preferred_name.lower() in name.lower():
            return i
    return None

def listen_for_name(mic_index=None):
    recognizer = sr.Recognizer()
    mic = sr.Microphone(device_index=mic_index) if mic_index is not None else sr.Microphone()
    attempts = 0
    while attempts < 3:
        speak("Say: Hello Parrot search for... and the person's name.")
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            print("[LISTENING]...")
            audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio).lower()
            print(f"[YOU SAID]: {command}")
            if "hello parrot search for" in command:
                name = command.split("hello parrot search for")[-1].strip()
                if name:
                    return name
            speak("Invalid format. Please try again.")
        except sr.UnknownValueError:
            speak("Could not understand.")
        except sr.RequestError:
            speak("Speech service error.")
        attempts += 1
    speak("Too many failed attempts.")
    return None

def recognize_and_follow(target_name, model_file="trained_faces.pkl"):
    if not os.path.exists(model_file):
        speak("Face data file not found.")
        return

    with open(model_file, "rb") as f:
        known_encodings, known_names = pickle.load(f)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        speak("Camera error.")
        return

    speak(f"Looking for {target_name}...")

    tracker = None
    tracking = False
    tolerance = 0.45
    found_once = False
    frame_count = 0
    last_face_encoding = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_small = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=0.25, fy=0.25), cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb_small)
        encodings = face_recognition.face_encodings(rgb_small, face_locs)

        found_face = False
        for (top, right, bottom, left), encoding in zip(face_locs, encodings):
            distances = face_recognition.face_distance(known_encodings, encoding)
            best = np.argmin(distances)
            if distances[best] < tolerance:
                name = known_names[best]
                if name.lower() == target_name.lower():
                    found_face = True
                    found_once = True
                    last_face_encoding = encoding
                    top, right, bottom, left = top*4, right*4, bottom*4, left*4
                    face_box = (left, top, right - left, bottom - top)

                    # Derive body box (expand vertically below face)
                    body_box = (
                        max(0, left - 40),
                        top,
                        min(1280 - left, (right - left) + 80),
                        min(720 - top, (bottom - top) * 3)
                    )

                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, body_box)
                    tracking = True
                    speak(f"{target_name.capitalize()} found. Starting body tracking.")
                    break

        if tracking and tracker is not None:
            success, box = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv2.putText(frame, f"Tracking {target_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            else:
                tracking = False
                tracker = None
                speak("Lost target. Re-scanning...")

        cv2.putText(frame, f"Target: {target_name.capitalize()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Parrot Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    speak("Session ended.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mic_index = select_microphone("earbuds")
    target = listen_for_name(mic_index)
    if target:
        recognize_and_follow(target)