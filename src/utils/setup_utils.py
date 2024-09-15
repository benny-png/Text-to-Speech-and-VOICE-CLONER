import os
import sys
import subprocess
from pathlib import Path

# Directory paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Adjusted to be outside src
FINETUNE_DIR = BASE_DIR / "src/utils/finetune-hf-vits"
UROMAN_DIR = BASE_DIR / "src/utils/uroman"
HF_TOKEN_PATH = Path.home() / ".huggingface/token"
REQUIREMENTS_PATH = BASE_DIR / "requirements.txt"

def install_python_requirements():
    """Install Python requirements for the project."""
    requirements_path = str(REQUIREMENTS_PATH).replace('\\', '\\\\')  # Escape backslashes for Windows
    command = f"{sys.executable} -m pip install -r {requirements_path}"
    print(f"Running command: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,  # Use shell=True to run the command as a string
            check=True,
            capture_output=True,
            text=True
        )
        print("Requirements installed successfully.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        print(f"Error output: {e.stderr}")
        print(f"Requirements file path: {REQUIREMENTS_PATH}")
        if not REQUIREMENTS_PATH.exists():
            print(f"Requirements file does not exist at {REQUIREMENTS_PATH}")
        raise

def setup_monotonic_align():
    """Set up monotonic alignment search for VITS/MMS finetuning."""
    monotonic_align_dir = FINETUNE_DIR / "monotonic_align"
    if not monotonic_align_dir.exists():
        raise FileNotFoundError(f"Directory not found: {monotonic_align_dir}")
    
    try:
        result = subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=str(monotonic_align_dir),
            check=True,
            capture_output=True,
            text=True
        )
        print("Monotonic align setup completed successfully.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error setting up monotonic align: {e}")
        print(f"Error output: {e.stderr}")
        raise

def setup_uroman():
    """Set the UROMAN environment variable."""
    if not UROMAN_DIR.exists():
        raise FileNotFoundError(f"Uroman directory not found: {UROMAN_DIR}")
    
    os.environ["UROMAN"] = str(UROMAN_DIR.resolve())
    print(f"UROMAN environment variable set to: {os.environ['UROMAN']}")
    return os.environ["UROMAN"]

def check_huggingface_cli_installed():
    """Check if Hugging Face CLI is installed."""
    try:
        result = subprocess.run(["huggingface-cli", "--version"], check=True, capture_output=True, text=True)
        print(f"Hugging Face CLI is installed. Version: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Hugging Face CLI is not installed.")
        return False

def install_huggingface_cli():
    """Install Hugging Face CLI using pip."""
    try:
        result = subprocess.run(["pip", "install", "huggingface_hub"], check=True, capture_output=True, text=True)
        print("Hugging Face CLI installed successfully.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error installing Hugging Face CLI: {e}")
        print(f"Error output: {e.stderr}")
        raise

def save_huggingface_token(token: str):
    """Save the Hugging Face token for future use."""
    if not HF_TOKEN_PATH.exists():
        HF_TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)

    HF_TOKEN_PATH.write_text(token)
    print(f"Hugging Face token saved to: {HF_TOKEN_PATH}")

def login_huggingface(token: str):
    """Log in to Hugging Face using the provided token."""
    try:
        print(f"Attempting to log in to Hugging Face with token: {token[:4]}...")  # Only print first 4 characters for security
        result = subprocess.run(
            ["huggingface-cli", "login", "--token", token],
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        print("Hugging Face CLI login command executed.")
        print(f"Command output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error logging in to Hugging Face: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("Starting setup process...")
    print(f"Base directory: {BASE_DIR}")
    print(f"Requirements file: {REQUIREMENTS_PATH}")
    print(f"Using Python interpreter: {sys.executable}")
    
    try:
        install_python_requirements()
        setup_monotonic_align()
        setup_uroman()
        
        if not check_huggingface_cli_installed():
            install_huggingface_cli()
        
        token = input("Please enter your Hugging Face token: ")
        save_huggingface_token(token)
        login_huggingface(token)
        
        print("Setup process completed successfully.")
    except Exception as e:
        print(f"Setup process failed: {e}")
        raise

if __name__ == "__main__":
    main()