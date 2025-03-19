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
import concurrent.futures
import threading

# Load environment variables from .env file
load_dotenv()

# Gemini API configuration
API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    API_KEY = "AIzaSyBo6NuG90H-FfdPpetru749VCn78Akvt6g"  # Replace with your API key if not using .env

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
SYSTEM_PROMPT = "You are a helpful robot assistant in a program designed for conversational interaction. Avoid using text formatting like markdown, hyphens, or asterisks as your responses will be spoken aloud. Keep responses concise and natural. The user says:"

# Voice recognition configuration
model_path = "../vosk-model-en-us-0.22"  # Update path to Vosk model
activation_word = input("Enter the activation word (default: 'hello'): ") or "hello"

# Initialisations
vosk_model = vosk.Model(model_path)
recognizer = vosk.KaldiRecognizer(vosk_model, 16000)
audio_queue = queue.Queue()

# Pygame for interruptible sound playback
pygame.mixer.init()

# Control flags and locks
activation_detected = False
stop_playback = False
is_speaking = False
audio_lock = threading.Lock()
state_lock = threading.Lock()

# Thread pool for the 3 cores
executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

# Chat history for context memory - with thread safety
chat_history_lock = threading.Lock()
chat_history = []

# TTS audio cache to reduce processing time for repeated responses
tts_cache = {}
tts_cache_lock = threading.Lock()

def speak(text):
    """Convert text to speech and play it"""
    global is_speaking, stop_playback
    
    # Check if this text is already in cache
    cache_key = text[:100]  # Use first 100 chars as key to avoid memory issues
    temp_filename = None
    
    with tts_cache_lock:
        if cache_key in tts_cache:
            temp_filename = tts_cache[cache_key]
            # Verify file still exists
            if not os.path.exists(temp_filename):
                del tts_cache[cache_key]
                temp_filename = None
    
    try:
        # Generate speech if not in cache
        if not temp_filename:
            tts = gTTS(text=text, lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                temp_filename = fp.name
                # Store in cache
                with tts_cache_lock:
                    # Limit cache size
                    if len(tts_cache) > 20:
                        # Remove oldest item
                        old_key, old_file = next(iter(tts_cache.items()))
                        if os.path.exists(old_file):
                            os.remove(old_file)
                        del tts_cache[old_key]
                    tts_cache[cache_key] = temp_filename

        with state_lock:
            is_speaking = True
            stop_playback = False
            
        with audio_lock:
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy() and not stop_playback:
                sleep(0.1)

    finally:
        with state_lock:
            is_speaking = False
        
        with audio_lock:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()  # Unload the file to free resources
        
        # Don't delete cached files

def callback(indata, frames, time, status):
    """Callback for audio stream to put audio data in queue"""
    audio_queue.put(bytes(indata))

def listen_for_activation():
    """Listen for the activation word"""
    global activation_detected, stop_playback
    
    # Reset recognizer to avoid previous speech contamination
    nonlocal_recognizer = vosk.KaldiRecognizer(vosk_model, 16000)
    
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16", channels=1, callback=callback):
        print("Listening for activation word...")
        while True:
            try:
                data = audio_queue.get(timeout=0.5)  # Add timeout to check stop flag periodically
                if nonlocal_recognizer.AcceptWaveform(data):
                    result = json.loads(nonlocal_recognizer.Result())
                    text = result.get("text", "").lower()
                    print(f"Recognized text: {text}")
                    if activation_word in text:
                        print(f"Activation word '{activation_word}' detected.")
                        with state_lock:
                            if is_speaking:
                                stop_playback = True
                                activation_detected = False
                            else:
                                activation_detected = True
                        return
            except queue.Empty:
                # Queue timeout occurred, just continue
                continue

def record_command():
    """Record and transcribe user command after activation"""
    # Flush the audio queue before recording a new command
    while not audio_queue.empty():
        audio_queue.get(block=False)
    
    # Create a fresh recognizer for this recording
    command_recognizer = vosk.KaldiRecognizer(vosk_model, 16000)
    
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16", channels=1, callback=callback):
        print("Listening...")
        silence_counter = 0
        max_silence = 20  # About 2 seconds of silence (assuming 10 callbacks per second)
        
        while True:
            try:
                data = audio_queue.get(timeout=0.1)
                if command_recognizer.AcceptWaveform(data):
                    result = json.loads(command_recognizer.Result())
                    text = result.get("text", "")
                    if text:
                        return text
                else:
                    partial = json.loads(command_recognizer.PartialResult())
                    if not partial.get("partial"):
                        silence_counter += 1
                    else:
                        silence_counter = 0
                    
                    # If long silence after speech, end recording
                    if silence_counter > max_silence:
                        final_result = json.loads(command_recognizer.FinalResult())
                        text = final_result.get("text", "")
                        return text if text else "I didn't catch that"
                    
            except queue.Empty:
                # Queue timeout occurred, just continue
                continue

def query_gemini(prompt):
    """Send a query to Gemini API with chat history for context"""
    
    # Add the current user message to chat history
    with chat_history_lock:
        chat_history.append({"role": "user", "parts": [{"text": prompt}]})
        # Create a copy of chat history for the request to avoid locks
        history_copy = chat_history.copy()

    # Prepare the request with system prompt and chat history
    data = {
        "contents": [
            {"role": "user", "parts": [{"text": SYSTEM_PROMPT}]},
            *history_copy
        ]
    }

    # Send request to Gemini API
    try:
        response = requests.post(GEMINI_URL, json=data, timeout=10)
        if response.status_code == 200:
            ai_response = response.json()['candidates'][0]['content']['parts'][0]['text']
            # Add AI response to chat history
            with chat_history_lock:
                chat_history.append({"role": "model", "parts": [{"text": ai_response}]})
                # Keep chat history to a reasonable size (last 10 exchanges)
                if len(chat_history) > 20:
                    chat_history = chat_history[-20:]
            return ai_response
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return "Sorry, I couldn't process that request."
    except requests.exceptions.Timeout:
        return "Sorry, the request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return "Sorry, there was a network error. Please check your connection."

def background_tts_preload():
    """Preload common TTS phrases in background"""
    common_phrases = ["Yes?", "I'm thinking...", "I didn't catch that", 
                     "Sorry, I couldn't process that request."]
    
    for phrase in common_phrases:
        tts = gTTS(text=phrase, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            with tts_cache_lock:
                tts_cache[phrase] = fp.name

def main_loop():
    """Main program loop with support for starting a new conversation on 'stop'"""
    global activation_detected, stop_playback, chat_history
    
    # Start background preloading of common TTS phrases
    executor.submit(background_tts_preload)
    
    while True:
        # Outer loop: wait for activation word to start a new conversation session
        print(f"Voice-enabled Gemini Chat started. Say the activation word '{activation_word}' to begin.")
        
        # Use a thread for listening so we can process other things in parallel
        activation_future = executor.submit(listen_for_activation)
        activation_future.result()  # Wait for activation
        
        speak("Yes?")
        
        # Inner loop: handle conversation until the user says "stop"
        while True:
            # Record command in the main thread
            user_input = record_command()
            print("User:", user_input)
            
            # If the user says "stop", clear history and break out of the inner loop
            if "stop" in user_input.lower():
                speak("Okay, stopping the conversation. Starting a new conversation.")
                print("Command 'stop' detected. Resetting conversation.")
                with chat_history_lock:
                    chat_history.clear()
                break

            # Use a thread from the pool to process the API request
            api_future = executor.submit(query_gemini, user_input)
            
            # Use another thread to say "I'm thinking..." if the API call takes too long
            def say_thinking():
                sleep(2)  # Wait 2 seconds before saying "I'm thinking..."
                with state_lock:
                    if not api_future.done():
                        speak("I'm thinking...")
            
            thinking_future = executor.submit(say_thinking)
            
            # Get the response
            response = api_future.result()
            print("AI:", response)
            
            # Cancel the "thinking" message if it hasn't started yet
            thinking_future.cancel()
            
            # Speak the response
            speak_future = executor.submit(speak, response)
            speak_future.result()
            
            print("Waiting for next command. Say 'stop' to end conversation.")

def cleanup():
    """Clean up resources"""
    # Clear the TTS cache files
    with tts_cache_lock:
        for filename in tts_cache.values():
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                except:
                    pass
    
    # Shut down the thread pool
    executor.shutdown(wait=False)
    
    # Close pygame
    pygame.mixer.quit()

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\nProgram terminated.")
    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to text mode...")
    finally:
        cleanup()

        