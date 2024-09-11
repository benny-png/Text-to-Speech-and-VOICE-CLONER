import os
import subprocess
from pathlib import Path

# Directory paths for finetuning and model checkpoints
BASE_DIR = Path(__file__).resolve().parent.parent
FINETUNE_DIR = BASE_DIR / "utils/finetune-hf-vits"
UROMAN_DIR = BASE_DIR / "utils/uroman"
HF_TOKEN_PATH = Path.home() / ".huggingface/token"


def install_python_requirements():
    """Install Python requirements for the finetune-hf-vits."""
    subprocess.run(["pip", "install", "-r", f"{FINETUNE_DIR}/requirements.txt"], check=True)


def setup_monotonic_align():
    """Set up monotonic alignment search for VITS/MMS finetuning."""
    subprocess.run(
        ["python", "setup.py", "build_ext", "--inplace"],
        cwd=f"{FINETUNE_DIR}/monotonic_align", check=True
    )


def setup_uroman():
    """Set the UROMAN environment variable."""
    if not UROMAN_DIR.exists():
        raise FileNotFoundError("Uroman directory not found in utils.")
    
    os.environ["UROMAN"] = str(UROMAN_DIR.resolve())
    return os.environ["UROMAN"]


def check_huggingface_cli_installed():
    """Check if Hugging Face CLI is installed."""
    try:
        subprocess.run(["huggingface-cli", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_huggingface_cli():
    """Install Hugging Face CLI using pip."""
    subprocess.run(["pip", "install", "huggingface_hub"], check=True)


def save_huggingface_token(token: str):
    """Save the Hugging Face token for future use."""
    if not HF_TOKEN_PATH.exists():
        HF_TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)

    HF_TOKEN_PATH.write_text(token)


def login_huggingface(token: str):
    """Log in to Hugging Face using the provided token."""
    subprocess.run(["huggingface-cli", "login"], input=token.encode(), check=True)
