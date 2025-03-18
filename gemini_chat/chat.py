import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Gemini API configuration
API_KEY = os.getenv('GOOGLE_API_KEY')
if not API_KEY:
    API_KEY = "YOUR_API_KEY_HERE"  # Replace with your API key if not using .env

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
SYSTEM_PROMPT = "You are a helpful robot assistant in a program designed for conversational interaction. Avoid using text formatting like markdown, hyphens, or asterisks as your responses will be spoken aloud. Keep responses concise and natural. The user says:"

# Chat history for context memory
chat_history = []

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
    try:
        response = requests.post(GEMINI_URL, json=data)
        response.raise_for_status()  # Raise exception for HTTP errors
        ai_response = response.json()['candidates'][0]['content']['parts'][0]['text']
        # Add AI response to chat history
        chat_history.append({"role": "model", "parts": [{"text": ai_response}]})
        return ai_response
    except Exception as e:
        print(f"Error: {e}")
        return "Sorry, I couldn't process that request."

def text_mode():
    """Text-only chat mode"""
    global chat_history
    print("Gemini Text Chat (Type 'exit' to quit)")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        response = query_gemini(user_input)
        print("AI:", response)

def try_voice_mode():
    """Attempt to initialize and run voice mode, with fallback to text mode"""
    try:
        import sounddevice as sd
        import queue
        import vosk
        import tempfile
        import pygame
        import threading
        from gtts import gTTS
        from time import sleep

        model_path = input("Enter the path to your Vosk model folder: ")
        if not os.path.exists(model_path):
            print(f"Model path '{model_path}' not found.")
            return False

        activation_word = input("Enter the activation word (default: 'jojo'): ") or "jojo"

        print(f"Initializing voice model from {model_path}...")
        vosk_model = vosk.Model(model_path)
        recognizer = vosk.KaldiRecognizer(vosk_model, 16000)
        audio_queue = queue.Queue()

        pygame.mixer.init()

        activation_detected = False
        stop_playback = False
        is_speaking = False

        def speak(text):
            nonlocal is_speaking, stop_playback
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
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)

        def callback(indata, frames, time, status):
            audio_queue.put(bytes(indata))

        def listen_for_activation():
            nonlocal activation_detected, stop_playback
            with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16", channels=1, callback=callback):
                while True:
                    data = audio_queue.get()
                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        text = result.get("text", "").lower()
                        if activation_word in text:
                            if is_speaking:
                                stop_playback = True
                                activation_detected = False
                            else:
                                activation_detected = True
                            return

        def record_command():
            with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16", channels=1, callback=callback):
                print("Listening...")
                while True:
                    data = audio_queue.get()
                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        text = result.get("text", "")
                        if text:
                            return text

        def main_voice_loop():
            nonlocal activation_detected, stop_playback
            print(f"Voice-enabled Gemini Chat started. Say the activation word '{activation_word}' to begin.")

            while True:
                print("Waiting for activation word...")
                listen_for_activation()

                if stop_playback:
                    stop_playback = False
                    activation_detected = True

                if activation_detected:
                    speak("Yes?")
                    activation_detected = False

                    user_input = record_command()
                    print("User:", user_input)

                    response = query_gemini(user_input)
                    print("AI:", response)

                    speak_thread = threading.Thread(target=speak, args=(response,))
                    speak_thread.start()

                    listen_thread = threading.Thread(target=listen_for_activation)
                    listen_thread.start()

                    speak_thread.join()
                    listen_thread.join()

        main_voice_loop()
        return True

    except Exception as e:
        print(f"Voice mode initialization failed: {e}")
        return False

if __name__ == "__main__":
    print("Gemini Chat System")
    print("==================")

    mode_choice = input("Choose mode (1 for Voice, 2 for Text): ")

    if mode_choice == "1":
        print("Initializing voice mode...")
        if not try_voice_mode():    
            print("Falling back to text mode...")
            text_mode()
    else:
        print("Starting text mode...")
        text_mode()