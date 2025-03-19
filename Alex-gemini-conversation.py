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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Gemini API configuration
API_KEY = os.getenv('Google API Key')  # Get API key from environment variables
if not API_KEY:
    API_KEY = "'Google API Key"  # Replace with your API key if not using .env
    
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
SYSTEM_PROMPT = "You are a helpful robot assistant in a program designed for conversational interaction. Avoid using text formatting like markdown, hyphens, or asterisks as your responses will be spoken aloud. Keep responses concise and natural. The user says:"

# Voice recognition configuration
model_path = r"C:\Users\Laptop-Alex\Downloads\vosk-model-en-us-0.22\vosk-model-en-us-0.22"  # Update path to Vosk model
activation_word = input("Enter the activation word (default: 'jojo'): ") or "jojo"

# Initialisations
vosk_model = vosk.Model(model_path)
recognizer = vosk.KaldiRecognizer(vosk_model, 16000)
audio_queue = queue.Queue()

# Pygame for interruptible sound playback
pygame.mixer.init()

# Control flags
activation_detected = False
stop_playback = False
is_speaking = False

# Chat history for context memory
chat_history = []

def speak(text):
    """Convert text to speech and play it"""
    global is_speaking, stop_playback
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            temp_filename = fp.name

        is_speaking = True
        stop_playback = False
        pygame.mixer.music.load(temp_filename)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy() and not stop_playback:
            sleep(0.1)

    finally:
        is_speaking = False
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()  # Datei entladen, um sie freizugeben
        sleep(0.5)  # Kurze Verzögerung, um sicherzustellen, dass die Datei nicht mehr gesperrt ist
        os.remove(temp_filename)  # Datei löschen


def callback(indata, frames, time, status):
    """Callback for audio stream to put audio data in queue"""
    audio_queue.put(bytes(indata))

def listen_for_activation():
    """Listen for the activation word"""
    global activation_detected, stop_playback
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16", channels=1, callback=callback):
        print("Listening for activation word...")  # Debugging message
        while True:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").lower()
                print(f"Recognized text: {text}")  # Debugging message
                if activation_word in text:
                    print(f"Activation word '{activation_word}' detected.")  # Debugging message
                    if is_speaking:
                        stop_playback = True
                        activation_detected = False
                    else:
                        activation_detected = True
                    return

def record_command():
    """Record and transcribe user command after activation"""
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16", channels=1, callback=callback):
        print("Listening...")  # Debugging message
        while True:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                if text:
                    return text

def query_gemini(prompt):
    """Send a query to Gemini API with chat history for context"""
    global chat_history
    
    # Add the current user message to chat history
    chat_history.append({"role": "user", "parts": [{"text": prompt}]})

    # Prepare the request with system prompt and chat history
    data = {
        "contents": [
            {"role": "user", "parts": [{"text": SYSTEM_PROMPT}]},
            *chat_history
        ]
    }
    
    # Send request to Gemini API
    response = requests.post(GEMINI_URL, json=data)
    if response.status_code == 200:
        ai_response = response.json()['candidates'][0]['content']['parts'][0]['text']
        # Add AI response to chat history
        chat_history.append({"role": "model", "parts": [{"text": ai_response}]})
        return ai_response
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return "Sorry, I couldn't process that request."

def main_loop():
    """Main program loop"""
    global activation_detected, stop_playback
    print(f"Voice-enabled Gemini Chat started. Say the activation word '{activation_word}' to begin.")
    
    # First, wait for activation word
    listen_for_activation()  # Wartet, bis Aktivierungswort erkannt wird
    speak("Yes?")

    while True:
        # Start listening for user input (no activation word needed now)
        user_input = record_command()  # Benutzerbefehl aufzeichnen
        if user_input.lower() == "stop":
            speak("Okay, stopping the conversation.")
            break  # Wenn 'stop' gesagt wird, beenden wir den Chat

        print("User:", user_input)
        
        # Send the user input to Gemini API and get response
        response = query_gemini(user_input)  
        print("AI:", response)
        
        # Speak the response
        speak(response)

        # Optionally, break loop after certain condition or continue.
        # In this case, we loop until user says "stop".

if __name__ == "__main__":
    try:
        # Start the main loop directly
        main_loop()
    except KeyboardInterrupt:
        pygame.mixer.quit()
        print("\nProgram terminated.")
    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to text mode...")
