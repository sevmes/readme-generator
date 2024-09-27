#!/usr/bin/python
import os
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting
import sys

if len(sys.argv) < 3:
    print("Usage: python main.py <PROJECT_ID> <LOCATION>")
    sys.exit(1)

# Project and location settings for Vertex AI
PROJECT_ID = sys.argv[1]
LOCATION = sys.argv[2]
MODEL_NAME = "gemini-1.5-pro-002"

# Configuration for text generation
GENERATION_CONFIG = {
    "max_output_tokens": 8192,
    "temperature": 0,
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

    code_files = {}
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
                    code_files[os.path.relpath(filepath)] = [content]
            except Exception as e:
                print(f"Error reading file {filepath}: {e}")
    return code_files


def send_message(chat, message, print_response=True):
    """Sends a message to the Gemini chat session with specified settings"""
    full_response = ""
    for response in chat._send_message_streaming(message, generation_config=GENERATION_CONFIG, safety_settings=SAFETY_SETTINGS):
        full_response += response.text
        if print_response:
            print(response.text, end="")
    if print_response:
        print("\n\n---------------\n\n")
    return full_response


def analyze_codebase(files):
    """Analyzes the provided code files using a multi-turn conversation with Gemini."""

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel(MODEL_NAME)
    chat = model.start_chat()

    initial_prompt = """
    Here is a code repository. Just respond "OK".
    Code files :
    """ + "".join([f"{filename} : \n{content}\n" for filename, content in files.items()])

    list_of_paths = "\n".join(files.keys())

    list_of_paths_prompt = f"""
        Here is the list of files in the project :
        {list_of_paths}
        Just respond with "OK".
    """
    
    send_message(chat, initial_prompt)
    send_message(chat, list_of_paths_prompt)
    generated_readme = send_message(chat, """
        Now write a description in two part of approximately the same length. It is a internal description that will help other internal developers understand the project.
        Focus on objective points and description instead of subjective thoughts such as why the project is well-written.
        The first part focuses on the purpose of the project, and should be business oriented. Be specific and get into the details of the business logic.
        The second part focuses on how the project works internally.
        Write your answer in french.
    """)

    with open("generated-README.md", "w+") as f:
        f.write(generated_readme)


if __name__ == "__main__":
    current_directory = os.getcwd()
    print(f"Current directory: {current_directory}")

    code_extensions = get_code_file_extensions(current_directory)
    code_files = read_code_files(current_directory, code_extensions)
    
    analyze_codebase(code_files)
