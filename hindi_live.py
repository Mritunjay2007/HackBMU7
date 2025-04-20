import cv2
import face_recognition
import speech_recognition as sr
import pickle
import numpy as np
import os
from indic_transliteration.sanscript import transliterate, DEVANAGARI, ITRANS
from gtts import gTTS
import pygame
import tempfile
import time
import re

# Initialize pygame mixer
pygame.mixer.init()

# TTS with multilingual support using gTTS and pygame Sound
def speak(text, lang="en"):
    print(f"[Komal-{lang.upper()}]: {text}")
    try:
        tts = gTTS(text=text, lang=lang)
        temp_path = os.path.join(tempfile.gettempdir(), f"komal_tts_{int(time.time() * 1000)}.mp3")
        tts.save(temp_path)
        sound = pygame.mixer.Sound(temp_path)
        sound.play()
        while pygame.mixer.get_busy():
            time.sleep(0.1)
        os.remove(temp_path)
    except Exception as e:
        print(f"[ERROR] Could not speak: {e}")

def select_microphone(preferred_name=None):
    mics = sr.Microphone.list_microphone_names()
    for i, name in enumerate(mics):
        print(f"[{i}] {name}")
        if preferred_name and preferred_name.lower() in name.lower():
            return i
    return None

def extract_hindi_name(command):
    print(f"[DEBUG Command]: {command}")
    trigger_phrases = ["कोमल", "komal"]
    action_phrases = ["खोजो", "khojo", "search for", "find"]
    for trigger in trigger_phrases:
        for action in action_phrases:
            if trigger in command and action in command:
                try:
                    name_part = command.split(trigger)[-1].split(action)[-1].strip()
                    name = name_part.split()[0].replace("को", "").strip()
                    return name
                except Exception:
                    pass
    return None

def transliterate_hindi_name(name_hi):
    try:
        return transliterate(name_hi, DEVANAGARI, ITRANS).lower()
    except Exception as e:
        print(f"[ERROR] in transliteration: {e}")
        return name_hi

def is_devanagari(text):
    return bool(re.search(r'[\u0900-\u097F]', text))

def listen_for_name(mic_index=None):
    recognizer = sr.Recognizer()
    mic = sr.Microphone(device_index=mic_index) if mic_index is not None else sr.Microphone()

    for attempt in range(3):
        speak("Say: Komal search for... or बोलिए: कोमल [नाम] को खोजो", "en")
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            print("[LISTENING]...")
            audio = recognizer.listen(source)

        try:
            command = recognizer.recognize_google(audio, language="en-IN").lower()
            print(f"[YOU SAID]: {command}")
            name_candidate = extract_hindi_name(command)
            if name_candidate:
                name_final = transliterate_hindi_name(name_candidate) if is_devanagari(name_candidate) else name_candidate
                print(f"[NAME]: {name_candidate} ➜ {name_final}")
                return name_final, "hi"
        except sr.UnknownValueError:
            pass
        except sr.RequestError:
            speak("Speech recognition service error.", "en")
            return None, None

        speak("Could not understand. Try again.", "en")

    speak("Too many failed attempts.", "en")
    return None, None

def recognize_and_follow(target_name, model_file="trained_faces.pkl"):
    if not os.path.exists(model_file):
        speak("Face data file not found. Please ensure trained_faces.pkl exists.")
        return

    with open(model_file, "rb") as f:
        known_encodings, known_names = pickle.load(f)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        speak("Camera error.")
        return

    speak(f"Looking for {target_name}...", "en")

    tracker = None
    tracking = False
    tolerance = 0.45

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
                    top, right, bottom, left = top*4, right*4, bottom*4, left*4
                    body_box = (
                        max(0, left - 40),
                        top,
                        min(1280 - left, (right - left) + 80),
                        min(720 - top, (bottom - top) * 3)
                    )
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, body_box)
                    tracking = True
                    speak(f"{target_name.capitalize()} found. Starting body tracking.", "en")
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
                speak("Lost target. Re-scanning...", "en")

        cv2.putText(frame, f"Target: {target_name.capitalize()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Komal Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    speak("Session ended.", "en")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mic_index = select_microphone("earbuds")
    target_name, lang = listen_for_name(mic_index)
    if target_name:
        speak(f"Searching for {target_name}...", lang)
        recognize_and_follow(target_name)