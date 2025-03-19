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
import RPi.GPIO as GPIO
import concurrent.futures
import gc

# GPIO einrichten
GPIO.setmode(GPIO.BCM)
GPIO.setup(12, GPIO.OUT)

# Qt GUI deaktivieren
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# CPU-Konfiguration: 3 von 4 Kernen für das Programm nutzen
TOTAL_CORES = multiprocessing.cpu_count()  # Sollte 4 für Raspberry Pi 4B sein
CORES_TO_USE = 3  # 3 Kerne für das Programm, 1 für das Betriebssystem

# Versuche, die CPU-Affinität zu setzen (wenn möglich)
try:
    import psutil
    # Aktuelle Prozess-ID
    p = psutil.Process(os.getpid())
    # Setze Affinität auf die ersten 3 Kerne (0, 1, 2)
    # Der vierte Kern (3) bleibt für das Betriebssystem
    p.cpu_affinity([0, 1, 2])
    print(f"CPU-Affinität auf Kerne 0,1,2 gesetzt. Kern 3 bleibt für das Betriebssystem.")
except (ImportError, AttributeError):
    print("CPU-Affinität konnte nicht gesetzt werden. Programm läuft ohne CPU-Beschränkung.")

# Kamara variablen - Verwende die tatsächlichen Pfade
video_devices = [
    "/dev/atlanta_01",
    "/dev/atlanta_02", 
    "/dev/atlanta_03",
    "/dev/atlanta_04",
    "/dev/atlanta_05"
]

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

# Optimierte Multithreading-Konfiguration für 3-Kern-Nutzung
face_detection_lock = threading.Lock()
speech_lock = threading.Lock()
api_lock = threading.Lock()

# Thread-Pool - optimiert für 3 Kerne
MAX_THREADS = CORES_TO_USE * 2  # 6 Threads für I/O-Operationen
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS)

# Prozess-Pool - optimiert für 3 Kerne
PROCESS_POOL_SIZE = CORES_TO_USE - 1  # 2 Prozesse für CPU-intensive Aufgaben
process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=PROCESS_POOL_SIZE)

# Verbesserte Performance-Einstellungen
NETWORK_TIMEOUT = 8  # Reduzierte Timeout-Zeit für Netzwerkanfragen
FRAME_RESIZE_FACTOR = 0.25  # Kleinere Frames für schnellere Verarbeitung
FACE_MATCHING_TOLERANCE = 0.6  # Gesichtserkennung Toleranz
CAMERA_SLEEP_TIME = 0.1  # Kurze Pause zwischen Kamera-Frames für höhere CPU-Auslastung
GARBAGE_COLLECTION_FREQ = 10  # Garbage Collection alle X Durchläufe

# Audio setup
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

# Performance-Counter für Statistiken
frame_counter = 0
detection_counter = 0
api_call_counter = 0

# Optimierte Funktion zum Laden eines einzelnen Gesichts
def load_single_face(image_path):
    """Laden eines einzelnen Gesichtsbildes - schnellere Verarbeitung"""
    try:
        name = os.path.splitext(os.path.basename(image_path))[0]
        image = face_recognition.load_image_file(image_path)
        
        # Reduziere Bildgröße für bessere Performance
        small_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        
        # Finde Gesichtsencodings
        encodings = face_recognition.face_encodings(small_image)
        
        # Speicher freigeben
        del image
        del small_image
        
        if encodings:
            return (encodings[0], name)
        else:
            print(f"Keine Gesichter in {name} gefunden")
            return None
    except Exception as e:
        print(f"Fehler beim Laden von {image_path}: {e}")
        return None

def load_known_faces():
    """Lädt bekannte Gesichter parallel mit dem Process Pool"""
    global known_encodings, known_names
    
    print("Lade bekannte Gesichter...")
    
    # Sammle alle Bilddateien
    image_paths = []
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            image_paths.append(image_path)
    
    print(f"Gefunden: {len(image_paths)} Gesichtsbilder")
    
    # Parallele Verarbeitung mit ProcessPoolExecutor
    results = []
    with process_pool as executor:
        # Starte alle Aufgaben
        futures = [executor.submit(load_single_face, path) for path in image_paths]
        
        # Sammle Ergebnisse
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Fehler bei Gesichtsladung: {e}")
    
    # Extrahiere Encodings und Namen
    known_encodings = [encoding for encoding, _ in results if encoding is not None]
    known_names = [name for _, name in results if name is not None]
    
    # Garbage Collection
    gc.collect()
    
    print(f"Erfolgreich geladen: {len(known_names)} Gesichter: {', '.join(known_names)}")

def speak(text):
    with speech_lock:
        try:
            # Thread-Pool für TTS-Generierung verwenden
            def generate_and_play():
                tts = gTTS(text=text, lang='en')
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    temp_filename = fp.name
                    tts.save(temp_filename)
                
                pygame.mixer.music.load(temp_filename)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    sleep(0.1)
                
                pygame.mixer.music.stop()
                os.remove(temp_filename)
            
            # Als separater Thread für bessere Parallelisierung
            thread_pool.submit(generate_and_play)
        except Exception as e:
            print(f"Fehler bei Sprachausgabe: {e}")

def callback(indata, frames, time, status):
    audio_queue.put(bytes(indata))

def record_command():
    global conversation_active
    recognizer.Reset()
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16", channels=1, callback=callback):
        print("Hörbereit...")
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
    global api_call_counter
    with api_lock:
        global chat_history
        
        chat_history.append({"role": "user", "parts": [{"text": prompt}]})
        data = {"contents": [{"role": "user", "parts": [{"text": SYSTEM_PROMPT + prompt}]}, *chat_history[-5:]]}
        
        try:
            # Netzwerkanfrage mit reduziertem Timeout für bessere Performance
            response = requests.post(GEMINI_URL, json=data, timeout=NETWORK_TIMEOUT)
            api_call_counter += 1
            
            if response.status_code == 200:
                ai_response = response.json()['candidates'][0]['content']['parts'][0]['text']
                chat_history.append({"role": "model", "parts": [{"text": ai_response}]})
                return ai_response
            else:
                print(f"API-Fehler: {response.status_code} - {response.text}")
                return "Entschuldigung, ich konnte diese Anfrage nicht verarbeiten."
        except Exception as e:
            print(f"Exception bei API-Aufruf: {e}")
            return "Entschuldigung, es gab ein Verbindungsproblem."

def get_greeting():
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        return "Guten Morgen"
    elif 12 <= current_hour < 18:
        return "Guten Tag"
    else:
        return "Guten Abend"

def listen_for_start():
    global conversation_active, last_detection_time
    while True:
        if not conversation_active and len(last_detection_time) > 0:
            user_input = record_command()
            if user_input and "start" in user_input.lower():
                print(f"Start erkannt! Starte Konversation mit anwesenden Benutzern: {', '.join(last_detection_time.keys())}")
                # Konversation als Thread mit hoher Priorität starten
                conversation_thread = threading.Thread(target=conversation_loop)
                conversation_thread.daemon = True
                conversation_thread.start()

def conversation_loop():
    global conversation_active, last_detection_time
    conversation_active = True
    present_users = ", ".join(last_detection_time.keys()) if last_detection_time else "unbekannten Benutzern"
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
        print("Benutzer:", user_input)
        
        # Verarbeite Anfrage mit Gemini
        response = query_gemini(user_input)
        
        print("AI:", response)
        speak(response)

def open_camera(device_path):
    """Versucht eine Kamera mit dem angegebenen Gerätepfad zu öffnen - optimiert für Geschwindigkeit"""
    try:
        # Direkter Versuch mit dem Gerätepfad
        cap = cv2.VideoCapture(device_path)
        if cap.isOpened():
            return cap
        
        # Versuch mit Index
        try:
            device_index = int(device_path.split('_')[-1]) - 1
            cap = cv2.VideoCapture(device_index)
            if cap.isOpened():
                return cap
        except ValueError:
            pass
        
        # Fallback-Methode: Erste verfügbare Kamera
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                return cap
        
        return None
    except Exception as e:
        print(f"Kamera-Fehler ({device_path}): {e}")
        return None

def process_camera_frame(device_path):
    """Verarbeitet einen einzelnen Frame einer Kamera - optimiert für Geschwindigkeit"""
    global frame_counter
    
    try:
        # Kamera öffnen
        cap = open_camera(device_path)
        if cap is None:
            return None
        
        # Frame lesen und sofort freigeben
        ret, frame = cap.read()
        cap.release()
        
        frame_counter += 1
        
        if not ret or frame is None or frame.size == 0:
            return None
            
        return frame
    except Exception:
        return None

def detect_faces_in_frame(frame):
    """Erkennt Gesichter in einem Frame - optimiert für Geschwindigkeit"""
    global detection_counter
    
    if frame is None:
        return []
    
    try:
        # Konvertiere zu RGB und reduziere Größe für schnellere Verarbeitung
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=FRAME_RESIZE_FACTOR, fy=FRAME_RESIZE_FACTOR)
        
        # Gesichtserkennung
        face_locations = face_recognition.face_locations(small_frame)
        
        if not face_locations:
            return []
        
        # Encodings für gefundene Gesichter
        scale_factor = int(1/FRAME_RESIZE_FACTOR)
        face_encodings = face_recognition.face_encodings(rgb_frame, 
            [(top*scale_factor, right*scale_factor, bottom*scale_factor, left*scale_factor) 
             for top, right, bottom, left in face_locations])
        
        detection_counter += len(face_encodings)
        
        # Speicher freigeben
        del rgb_frame
        del small_frame
        
        return face_encodings
    except Exception as e:
        print(f"Fehler bei Gesichtserkennung: {e}")
        return []

def compare_faces_with_known(face_encodings):
    """Vergleicht gefundene Gesichter mit bekannten Gesichtern - optimiert für Geschwindigkeit"""
    if not face_encodings or not known_encodings:
        return []
    
    matches = []
    for face_encoding in face_encodings:
        # Vergleiche mit bekannten Gesichtern
        face_matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=FACE_MATCHING_TOLERANCE)
        if any(face_matches):
            try:
                match_index = face_matches.index(True)
                matches.append(known_names[match_index])
            except (ValueError, IndexError):
                pass
    
    return matches

def process_single_camera(device_path):
    """Verarbeitet eine einzelne Kamera und gibt erkannte Gesichter zurück - optimiert für Geschwindigkeit"""
    try:
        # Frame holen und verarbeiten
        frame = process_camera_frame(device_path)
        if frame is None:
            return []
        
        # Gesichter erkennen
        face_encodings = detect_faces_in_frame(frame)
        if not face_encodings:
            return []
        
        # Gesichter vergleichen
        matches = compare_faces_with_known(face_encodings)
        
        # Speicher freigeben
        del frame
        del face_encodings
        
        return matches
    except Exception:
        return []

def print_performance_stats():
    """Gibt Performance-Statistiken aus"""
    global frame_counter, detection_counter, api_call_counter
    print(f"\n--- PERFORMANCE STATISTIKEN ---")
    print(f"Verarbeitete Frames: {frame_counter}")
    print(f"Erkannte Gesichter: {detection_counter}")
    print(f"API-Aufrufe: {api_call_counter}")
    print(f"CPU-Nutzung: {CORES_TO_USE}/{TOTAL_CORES} Kerne")
    print(f"Thread-Pool: {MAX_THREADS} Threads")
    print(f"Process-Pool: {PROCESS_POOL_SIZE} Prozesse")
    print(f"Aktuelle Zeit: {datetime.now()}")
    print(f"------------------------------\n")
    
    # Zähler zurücksetzen
    frame_counter = 0
    detection_counter = 0
    api_call_counter = 0

def face_detection():
    """Hauptschleife für die Gesichtserkennung - maximale Performance auf 3 Kernen"""
    try:
        print(f"Starte Gesichtserkennung mit {CORES_TO_USE} von {TOTAL_CORES} Kernen...")
        print("Alle 3 Kerne können bis zu 100% ausgelastet werden.")
        
        # Kameras sequentiell aber sehr schnell verarbeiten
        camera_index = 0
        stats_counter = 0
        
        while True:            
            # Verarbeite eine Kamera pro Durchlauf - maximale Geschwindigkeit
            device = video_devices[camera_index]
            matches = process_single_camera(device)
            
            if matches:
                print(f"Kamera {device}: Erkannte Gesichter: {', '.join(matches)}")
            
            # Nächste Kamera
            camera_index = (camera_index + 1) % len(video_devices)
            
            # Zustandsaktualisierung bei jedem kompletten Durchlauf
            if camera_index == 0:
                # Alle Kameras parallel scannen für vollständige Erkennung
                all_matches = []
                
                # Thread-Pool für parallele Kamera-Verarbeitung
                future_to_camera = {thread_pool.submit(process_single_camera, dev): dev for dev in video_devices}
                for future in concurrent.futures.as_completed(future_to_camera):
                    matches = future.result()
                    all_matches.extend(matches)
                
                # Aktualisiere erkannte Benutzer
                current_detections = {}
                for name in all_matches:
                    current_detections[name] = datetime.now()
                
                # Aktualisiere den Zustand und steuere GPIO
                with face_detection_lock:
                    global last_detection_time
                    last_detection_time = current_detections
                    
                    # GPIO-Steuerung
                    if current_detections:
                        detected_names = ", ".join(current_detections.keys())
                        print(f"[{datetime.now()}] Bekannte Gesichter erkannt: {detected_names} - Anton mocht pause")
                        GPIO.output(12, GPIO.HIGH)
                    else:
                        print(f"[{datetime.now()}] Keine bekannten Gesichter erkannt - Anton fetzt ob")
                        GPIO.output(12, GPIO.LOW)
                
                # Garbage Collection periodisch ausführen
                stats_counter += 1
                if stats_counter % GARBAGE_COLLECTION_FREQ == 0:
                    gc.collect()
                    print_performance_stats()
            
            # Minimale Pause für maximale CPU-Auslastung auf 3 Kernen
            sleep(CAMERA_SLEEP_TIME)
            
    except Exception as e:
        print(f"Fehler in der Hauptschleife: {e}")
    finally:
        GPIO.output(12, GPIO.LOW)
        GPIO.cleanup()

if __name__ == "__main__":
    try:
        print(f"\n--- JOJO SYSTEM START ---")
        print(f"Optimiert für Raspberry Pi: Nutzung von {CORES_TO_USE} von {TOTAL_CORES} CPU-Kernen")
        print(f"Maximale Performance-Einstellung aktiviert!")
        print(f"----------------------------\n")
        
        multiprocessing.freeze_support()
        
        # Bekannte Gesichter parallel laden mit Prozess-Pool
        load_known_faces()
        
        # Höre auf Startbefehle (niedriger Thread)
        start_listener_thread = threading.Thread(target=listen_for_start, daemon=True)
        start_listener_thread.start()
        
        # Starte Gesichtserkennung als Hauptprozess
        face_detection()
        
    except KeyboardInterrupt:
        print("\nProgramm wird beendet...")
    finally:
        # Ressourcen freigeben
        thread_pool.shutdown()
        process_pool.shutdown()
        pygame.mixer.quit()
        GPIO.cleanup()
        print("\nProgramm beendet.")