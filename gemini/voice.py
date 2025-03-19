import os
import sounddevice as sd
import queue
import vosk
import requests
import json
from gtts import gTTS
import tempfile
import pygame
from time import sleep
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Gemini API configuration
API_KEY = os.getenv('AIzaSyBo6NuG90H-FfdPpetru749VCn78Akvt6g')
if not API_KEY:
    API_KEY = "AIzaSyBo6NuG90H-FfdPpetru749VCn78Akvt6g"  # Replace with your API key if not using .env

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
SYSTEM_PROMPT = "You are a helpful robot assistant in a program designed for conversational interaction. Avoid using text formatting like markdown, hyphens, or asterisks as your responses will be spoken aloud. Keep responses concise and natural. The user says:"

# Voice recognition configuration
model_path = r"C:\Users\Laptop-Alex\Downloads\vosk-model-en-us-0.22\vosk-model-en-us-0.22"  # Passe den Pfad an
activation_word = input("Enter the activation word (default: 'hello'): ") or "hello"

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
        pygame.mixer.music.unload()  # Unload the file to free resources
        sleep(0.5)  # Brief delay to ensure file is not locked
        os.remove(temp_filename)  # Remove the temporary file

def callback(indata, frames, time, status):
    """Callback for audio stream to put audio data in queue"""
    audio_queue.put(bytes(indata))

def listen_for_activation():
    """Listen for the activation word"""
    global activation_detected, stop_playback
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16", channels=1, callback=callback):
        print("Listening for activation word...")
        while True:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").lower()
                print(f"Recognized text: {text}")
                if activation_word in text:
                    print(f"Activation word '{activation_word}' detected.")
                    if is_speaking:
                        stop_playback = True
                        activation_detected = False
                    else:
                        activation_detected = True
                    return

def record_command():
    """Record and transcribe user command after activation"""
    # Flush the audio queue before recording a new command
    while not audio_queue.empty():
        audio_queue.get()
        
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16", channels=1, callback=callback):
        print("Listening...")
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
    """Main program loop with support for starting a new conversation on 'stop'"""
    global activation_detected, stop_playback, chat_history
    while True:
        # Outer loop: wait for activation word to start a new conversation session
        print(f"Voice-enabled Gemini Chat started. Say the activation word '{activation_word}' to begin.")
        listen_for_activation()
        speak("Yes?")
        
        # Inner loop: handle conversation until the user says "stop"
        while True:
            user_input = record_command()
            print("User:", user_input)
            
            # If the user says "stop", clear history and break out of the inner loop
            if "stop" in user_input.lower():
                speak("Okay, stopping the conversation. Starting a new conversation.")
                print("Command 'stop' detected. Resetting conversation.")
                chat_history.clear()
                break

            response = query_gemini(user_input)
            print("AI:", response)
            speak(response)

            print("Waiting for next command. Say 'stop' to end conversation.")

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        pygame.mixer.quit()
        print("\nProgram terminated.")
    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to text mode...")


        
