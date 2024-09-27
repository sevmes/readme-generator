import os
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting

# Project and location settings for Vertex AI
PROJECT_ID = "morini-733-20240208103453"
LOCATION = "us-central1"
MODEL_NAME = "gemini-pro-experimental"

# Configuration for text generation
GENERATION_CONFIG = {
    "max_output_tokens": 8192,
    "temperature": 0.2,
    "top_p": 0.95,
}

# Safety settings for the model
SAFETY_SETTINGS = [
    SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE),
    SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE),
    SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE),
    SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE),
]

# Excluded directories and file extensions for code analysis
EXCLUDED_DIRECTORIES = [".git", "genproto", ".venv", "node_modules", "dist", "build", "tests", "test", "bin", "target", "out", ".next"]


def get_code_file_extensions(directory):
    """Identifies code file extensions within a directory using Gemini."""

    found_extensions = {os.path.splitext(filename)[1][1:] for root, _, filenames in os.walk(directory) for filename in filenames}

    prompt = f"""
    Please return a list of extensions that identify files containing source code.
    The format should be a list of extension separated by commas. Your response should only
    be the list of extension, or ".c" if no extension in the original list is an extension of source code file.
    Example input : 
    csv,ts,jpg,cpp,java
    Output :
    ts,c,cpp,java

    Here is the input :
    {",".join(found_extensions)}
    """

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel(MODEL_NAME)
    chat = model.start_chat()
    response = chat.send_message(prompt)
    print(f"Code extensions found : {response.text}")
    return response.text.split(",")


def read_code_files(directory, allowed_extensions):
    """Reads code files from the specified directory, excluding specified subdirectories."""

    code_files = []
    for root, _, filenames in os.walk(directory):
        # Load .gitignore for the current directory and its parents
        gitignore_entries = []
        current_dir = root
        while current_dir != "/":
            gitignore_path = os.path.join(current_dir, ".gitignore")
            if os.path.exists(gitignore_path):
                with open(gitignore_path, "r") as f:
                    gitignore_entries.extend(f.read().splitlines())
            current_dir = os.path.dirname(current_dir)
        
        for filename in filenames:
            if not any(filename.endswith(ext) for ext in allowed_extensions):
                continue

            filepath = os.path.join(root, filename)

            # Check if file path should be ignored based on .gitignore
            if any(f"/{entry.strip()}/" in filepath for entry in gitignore_entries):
                print(f"Skipping ignored file: {filepath}")
                continue
            
            if any(f"/{subdir}/" in filepath for subdir in EXCLUDED_DIRECTORIES):
                continue

            print(f"Reading file: {filepath}")
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    code_files.append(f"File path : {filepath}\n```\n{content}\n```")
            except Exception as e:
                print(f"Error reading file {filepath}: {e}")
    return code_files



def send_message(chat, message, print_response=True):
    """Sends a message to the Gemini chat session with specified settings"""
    for response in chat._send_message_streaming(message, generation_config=GENERATION_CONFIG, safety_settings=SAFETY_SETTINGS):
        if print_response:
            print(response.text, end="")
    if print_response:
        print("\n\n---------------\n\n")


def analyze_codebase(files):
    """Analyzes the provided code files using a multi-turn conversation with Gemini."""

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel(MODEL_NAME)
    chat = model.start_chat()

    initial_prompt = """
    Here is a code repository. Just respond "OK".
    Code files :
    """ + "".join(files)


    send_message(chat, initial_prompt)
    send_message(chat, "Give the full project structure.")
    send_message(chat, """
        Now write a description in two part of approximately the same length.
        The first part is focuses on the purpose of the project. Be specific and get into the details, but keep it business oriented and not technical.
        The second part focuses on how the project works internally.
        This is a internal description that will help other developers understand the project. It is not meant to advertise
        how good and how cool it is. It's about having the details of what it is and how it works. Write it in french.
    """)

    while True:
        user_input = input("Enter a new prompt (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        send_message(chat, user_input)

    chat.end_chat()

    with open("response.txt", "w") as f:
        f.write(chat.last_response.text)

    print(chat.last_response.text)


if __name__ == "__main__":
    current_directory = os.getcwd()
    print(f"Current directory: {current_directory}")

    code_extensions = get_code_file_extensions(current_directory)
    code_files = read_code_files(current_directory, code_extensions)
    
    analyze_codebase(code_files)
