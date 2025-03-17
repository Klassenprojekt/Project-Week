import os
import google.genai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment variable
api_key = os.getenv('GOOGLE_API_KEY')

if not api_key:
    raise ValueError("API key is missing. Please set the GOOGLE_API_KEY environment variable.")

# Configure the client with the API key
client = genai.Client(api_key=api_key)

# Define the model name
MODEL_NAME = "gemini-2.0-flash"  # Ensure this model is available in your project

# Function to generate content
def get_gemini_response(prompt):
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error: {e}"

# Main function to interact with the chatbot
def main():
    print("Gemini Chatbot (Type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        print("Gemini:", get_gemini_response(user_input))

if __name__ == "__main__":
    main()
