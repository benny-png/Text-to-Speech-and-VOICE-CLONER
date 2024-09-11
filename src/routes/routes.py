from fastapi import FastAPI, Form, UploadFile, File
import subprocess
from typing import List
from pathlib import Path
from ..utils.setup_utils import (
    install_python_requirements, setup_monotonic_align, setup_uroman,
    check_huggingface_cli_installed, install_huggingface_cli,
    save_huggingface_token, login_huggingface
)



HF_TOKEN_PATH = Path.home() / ".huggingface/token"

app = FastAPI()



@app.get("/")
async def read_root():
    return {"message": "Welcome to the Voice Cloner API"}




@app.post("/setup")
async def install_requirements(token: str = Form(...)):
    """Install requirements and set up necessary files, Hugging Face login, and UROMAN setup."""
    try:
        # Step 1: Install Python requirements for the finetune-hf-vits
        install_python_requirements()

        # Step 2: Set up monotonic alignment search for VITS/MMS finetuning
        setup_monotonic_align()

        # Step 3: Set up UROMAN
        uroman_path = setup_uroman()
        print(f"UROMAN environment variable set to: {uroman_path}")

        # Step 4: Check if Hugging Face CLI is installed
        if not check_huggingface_cli_installed():
            print("Hugging Face CLI not found, installing...")
            install_huggingface_cli()

        # Step 5: Save the Hugging Face token for future use
        save_huggingface_token(token)
        print(f"Hugging Face token saved to: {HF_TOKEN_PATH}")

        # Step 6: Log in to Hugging Face using the provided token
        login_huggingface(token)
        print("Hugging Face login successful.")

        return {"status": "Success", "message": "Dependencies installed, Hugging Face login completed, and UROMAN set up."}
    
    except subprocess.CalledProcessError as e:
        return {"status": "Error", "message": f"An error occurred: {str(e)}"}



@app.post("/finetune")
async def finetune_model(
    dataset: UploadFile = File(...),
    model_type: str = Form(...),
    language: str = Form(default="en")
):
    """
    Fine-tune a VITS or MMS model.
    Args:
        dataset: Uploaded dataset for fine-tuning (80 to 150 samples).
        model_type: 'vits' for VITS-based models, 'mms' for MMS models.
        language: Language code for MMS model, e.g., "en", "es".
    """
    
    return "Not Immplemented"


@app.post("/generate-audio")
async def generate_audio(text: str = Form(...), model_type: str = Form(...)):
    """
    Generate audio from text using the finetuned VITS or MMS model.
    Args:
        text: The text to be converted to speech.
        model_type: 'vits' or 'mms' for the appropriate model.
    """
    return "Not Implemented"