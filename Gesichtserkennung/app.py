import os
import sounddevice as sd
import queue
import vosk
import requests
import json
import os
import sounddevice as sd
import queue
import vosk
import requests
import json
from gtts import gTTS
import tempfile
import pygame
import threading
from time import sleep
import face_recognition
import cv2
from datetime import datetime
import multiprocessing
import RPi.GPIO as GPIO  # Raspberry Pi GPIO-Modul

# GPIO einrichten
GPIO.setmode(GPIO.BCM)
GPIO.setup(12, GPIO.OUT)

# Qt GUI deaktivieren
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Global variables
model_path = "/home/anton/Voic_model/vosk-model-en-us-0.22/vosk-model-en-us-0.22"
API_KEY = "AIzaSyDd5x5Xc6potNayl0BDkhxe7B2YN2iuvyc"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
SYSTEM_PROMPT = "You are a Helpful Robot called JOJO in a Program which is made for speaking with other people so it is given to use shortend answers however when the user wants a specific answer u can make a 20+ word text. However do not use any sort of text enhancer for example markdown because your prompt is directly spoken out and do not use any - and *. The User Prompt is:"

conversation_active = False
last_conversation_end = None
CONVERSATION_COOLDOWN = 30
activation_detected = False
user_cooldowns = {}

vosk_model = vosk.Model(model_path)
recognizer = vosk.KaldiRecognizer(vosk_model, 16000)
audio_queue = queue.Queue()
pygame.mixer.init()

chat_history = []

KNOWN_FACES_DIR = "known_faces"
known_encodings = []
known_names = []
current_user = None

last_detection_time = {}
last_greeting_time = {}
tracking_status = {}

def load_known_faces():
    print("Loading known faces...")
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                name = os.path.splitext(filename)[0]
                known_names.append(name)
    print("Faces loaded!")

def speak(text):
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            temp_filename = fp.name

        pygame.mixer.music.load(temp_filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            sleep(0.1)
    finally:
        pygame.mixer.music.stop()
        os.remove(temp_filename)

def callback(indata, frames, time, status):
    audio_queue.put(bytes(indata))

def record_command():
    global conversation_active
    recognizer.Reset()
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16", channels=1, callback=callback):
        print("Listening...")
        while True:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").lower()
                if text:
                    if "stop" in text:
                        conversation_active = False
                        return None
                    return text

def query_gemini(prompt):
    global chat_history
    chat_history.append({"role": "user", "parts": [{"text": prompt}]})
    data = {"contents": [{"role": "user", "parts": [{"text": SYSTEM_PROMPT}]}, *chat_history]}
    response = requests.post(GEMINI_URL, json=data)
    if response.status_code == 200:
        ai_response = response.json()['candidates'][0]['content']['parts'][0]['text']
        chat_history.append({"role": "model", "parts": [{"text": ai_response}]})
        return ai_response
    else:
        return "Sorry, I couldn't process that."

def get_greeting():
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        return "Good morning"
    elif 12 <= current_hour < 18:
        return "Good day"
    else:
        return "Good evening"

def listen_for_start():
    global conversation_active, last_detection_time
    while True:
        if not conversation_active and len(last_detection_time) > 0:
            user_input = record_command()
            if user_input and "start" in user_input.lower():
                print(f"Start detected! Starting conversation with present users: {', '.join(last_detection_time.keys())}")
                conversation_thread = threading.Thread(target=conversation_loop)
                conversation_thread.start()

def conversation_loop():
    global conversation_active, last_detection_time
    conversation_active = True
    present_users = ", ".join(last_detection_time.keys())
    speak(f"Starting conversation with {present_users}. How can I help you today?")
    while conversation_active:
        if len(last_detection_time) == 0:
            speak("I don't see anyone anymore. Ending conversation.")
            conversation_active = False
            break
        user_input = record_command()
        if user_input is None or "start" in user_input.lower():
            speak("Goodbye! Let me know if you need anything else.")
            conversation_active = False
            break
        print("User:", user_input)
        response = query_gemini(user_input)
        print("AI:", response)
        speak(response)

# Überarbeitete face_detection Funktion für headless Betrieb
def face_detection():
    video_capture = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Check if any detected face matches a known face
            known_face_detected = False
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
                if any(matches):
                    known_face_detected = True
                    break
            
            # Only set GPIO high when a known face is detected
            if known_face_detected:
                print(f"[{datetime.now()}] Known face detected - Anton mocht pause")
                GPIO.output(12, GPIO.HIGH)
            else:
                print(f"[{datetime.now()}] No known face detected - Anton fetzt ob")
                GPIO.output(12, GPIO.LOW)

            sleep(0.1)
    finally:
        video_capture.release()
        GPIO.output(12, GPIO.LOW)
        GPIO.cleanup()


if __name__ == "__main__":
    try:
        multiprocessing.freeze_support()
        load_known_faces()

        start_listener_thread = threading.Thread(target=listen_for_start, daemon=True)
        start_listener_thread.start()

        face_detection()  # Gesichtserkennung headless starten

    except KeyboardInterrupt:
        pygame.mixer.quit()
        GPIO.cleanup()
        print("\nProgram terminated.")