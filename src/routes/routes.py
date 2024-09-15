from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from transformers import VitsModel, AutoTokenizer
import subprocess
from typing import List
from pathlib import Path
import shutil
import json
import scipy.io.wavfile
from pydantic import BaseModel
import torch
import re
from typing import Optional
import logging
import psutil
import os
from ..utils.audio_dataset_stuff.dataset_prep import create_and_upload_dataset  # Import the dataset module we created earlier
from ..utils.setup_utils import (
    install_python_requirements, setup_monotonic_align, setup_uroman,
    check_huggingface_cli_installed, install_huggingface_cli,
    save_huggingface_token, login_huggingface
)

# Constants
HF_TOKEN_PATH = Path.home() / ".huggingface/token"
os.environ["WANDB_MODE"] = "disabled"

# Initialize the FastAPI app
app = FastAPI(title="Voice Cloner API")

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize global model and tokenizer
model = None
tokenizer = None

@app.on_event("startup")
async def startup_event():
    """
    Startup event to load the heavy dependencies once.
    """
    global model, tokenizer
    logger.info("Loading the model and tokenizer on startup...")
    model = VitsModel.from_pretrained("YOUR_MODEL_NAME")
    tokenizer = AutoTokenizer.from_pretrained("YOUR_MODEL_NAME")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event to cleanup during shutdown.
    """
    global model, tokenizer
    del model
    del tokenizer
    logger.info("Cleaning up resources on shutdown...")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Voice Cloner API"}

@app.post("/setup")
async def install_requirements(token: str = Form(...)):
    """Install requirements and set up necessary files, Hugging Face login, and UROMAN setup."""
    try:
        logger.info("Starting setup process...")

        # Step 1: Install Python requirements for the finetune-hf-vits
        logger.info("Installing Python requirements...")
        install_python_requirements()

        # Step 2: Set up monotonic alignment search for VITS/MMS finetuning
        logger.info("Setting up monotonic alignment...")
        setup_monotonic_align()

        # Step 3: Set up UROMAN
        logger.info("Setting up UROMAN...")
        uroman_path = setup_uroman()
        logger.info(f"UROMAN environment variable set to: {uroman_path}")

        # Step 4: Check if Hugging Face CLI is installed
        logger.info("Checking Hugging Face CLI installation...")
        if not check_huggingface_cli_installed():
            logger.info("Hugging Face CLI not found, installing...")
            install_huggingface_cli()

        # Step 5: Save the Hugging Face token for future use
        logger.info("Saving Hugging Face token...")
        save_huggingface_token(token)
        logger.info(f"Hugging Face token saved to: {HF_TOKEN_PATH}")

        # Step 6: Log in to Hugging Face using the provided token
        logger.info("Attempting Hugging Face login...")
        login_success = login_huggingface(token)
        if not login_success:
            raise HTTPException(status_code=400, detail="Hugging Face login failed")
        logger.info("Hugging Face login successful.")

        return {"status": "Success", "message": "Dependencies installed, Hugging Face login completed, and UROMAN set up."}
    
    except Exception as e:
        logger.error(f"Error during setup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during setup: {str(e)}")

@app.post("/finetune")
async def finetune_model(
    hf_dataset_repo: str = Form(None),
    dataset_dir: str = Form(None),
    csv_file: UploadFile = File(None),  # Allow None for optional CSV file
    model_type: str = Form(default="mms"),
    voice_name: str = Form(...),
    language: str = Form(default="sw")
):
    """
    Fine-tune a VITS or MMS model based on the provided dataset, model type, voice name, and language.
    
    Args:
        hf_dataset_repo: Hugging Face dataset repository if the data is already uploaded.
        dataset_dir: Path to the directory containing .wav files (optional if hf_dataset_repo is provided).
        csv_file: Uploaded CSV file containing text data (optional if hf_dataset_repo is provided).
        model_type: 'vits' or 'mms'
        voice_name: Custom voice name for the fine-tuning model
        language: Language code for the model, default 'sw' for Swahili
    """
    try:
        if not check_huggingface_cli_installed():
            install_huggingface_cli()
            
        token = None
        with open(HF_TOKEN_PATH, 'r') as token_file:
            token = token_file.read().strip()
        if not token:
            return {"status": "Error", "message": "No Hugging Face token found. Please log in first."}
        login_huggingface(token)

        # Memory check before proceeding
        mem = psutil.virtual_memory()
        logger.info(f"Available memory before fine-tuning: {mem.available} bytes")

        if hf_dataset_repo:
            repo_id = hf_dataset_repo
            logger.info(f"Using Hugging Face dataset repo: {hf_dataset_repo}")
        else:
            if not dataset_dir or not csv_file:
                raise HTTPException(status_code=400, detail="Either hf_dataset_repo or both dataset_dir and csv_file must be provided.")
            
            if csv_file:
                csv_path = Path(f"src/files/datasets/{voice_name}_metadata.csv")
                with open(csv_path, "wb") as f:
                    shutil.copyfileobj(csv_file.file, f)

            repo_id = f"Benjamin-png/dataset_{model_type}_{language}_{voice_name}"
            create_and_upload_dataset(dataset_dir, str(csv_path), repo_id)

        hub_model_id = f"{model_type}-tts-{language}-{voice_name}-finetuned"
        subprocess.run(["huggingface-cli", "repo", "create", hub_model_id, "--yes"], check=True)

        checkpoints = {
            "en": "ylacombe/mms-tts-eng-train",
            "sw": "Benjamin-png/swa-checkpoint",
            "ko": "ylacombe/mms-tts-kor-train",
            "mr": "ylacombe/mms-tts-mar-train",
            "ta": "ylacombe/mms-tts-tam-train",
            "gu": "ylacombe/mms-tts-guj-train"
        }
        checkpoint = checkpoints.get(language.lower(), "Benjamin-png/swa-checkpoint")

        config = {
            "project_name": f"{voice_name}_vocals",
            "push_to_hub": True,
            "hub_model_id": hub_model_id,
            "overwrite_output_dir": True,
            "output_dir": f"src/files/outputs/{voice_name}_tts_finetuned",
            "dataset_name": repo_id,
            "audio_column_name": "audio",
            "text_column_name": "text",
            "train_split_name": "train",
            "eval_split_name": "train",
            "speaker_id_column_name": "speaker_id",
            "override_speaker_embeddings": True,
            "filter_on_speaker_id": 1,
            "max_duration_in_seconds": 20,
            "min_duration_in_seconds": 1.0,
            "max_tokens_length": 500,
            "model_name_or_path": checkpoint,
            "preprocessing_num_workers": 4,
            "do_train": True,
            "num_train_epochs": 200,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": False,
            "per_device_train_batch_size": 16,
            "learning_rate": 2e-5,
            "adam_beta1": 0.8,
            "adam_beta2": 0.99,
            "warmup_ratio": 0.01,
            "group_by_length": False,
            "do_eval": True,
            "eval_steps": 50,
            "per_device_eval_batch_size": 16,
            "max_eval_samples": 25,
            "do_step_schedule_per_epoch": True,
            "weight_disc": 3,
            "weight_fmaps": 1,
            "weight_gen": 1,
            "weight_kl": 1.5,
            "weight_duration": 1,
            "weight_mel": 35,
            "fp16": torch.cuda.is_available(),
            "seed": 456
        }

        config_folder = Path(f"src/files/config/{voice_name}")
        config_folder.mkdir(parents=True, exist_ok=True)
        config_path = config_folder / f"{voice_name}_config.json"

        with open(config_path, "w") as config_file:
            json.dump(config, config_file, indent=4)

        subprocess.run(
            ["accelerate", "launch", "src/utils/finetune-hf-vits/run_vits_finetuning.py", str(config_path)],
            check=True
        )

        mem = psutil.virtual_memory()
        logger.info(f"Available memory after fine-tuning: {mem.available} bytes")

        return {"status": "Success", "message": f"Model fine-tuned successfully and uploaded to {hub_model_id}"}
    except subprocess.CalledProcessError as e:
        logger.error(f"Subprocess error: {str(e)}")
        return {"status": "Error", "message": f"An error occurred: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"status": "Error", "message": f"An error occurred: {str(e)}"}

class AudioGenerationRequest(BaseModel):
    text: str
    filename: str
    model_name: str

def sanitize_filename(filename: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
    return sanitized[:50]

@app.post("/generate_audio")
async def generate_audio(request: AudioGenerationRequest):
    """
    Generate speech from text using the specified TTS model.

    Args:
        request: Contains the input text for audio generation, the desired filename, and the model name.
        
    Returns:
        The path of the generated audio file.
    """
    text = request.text
    filename = request.filename
    model_name = request.model_name

    sanitized_filename = sanitize_filename(filename)
    audio_file_path = f"src/files/outputs/{sanitized_filename}_tts.wav"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        global model, tokenizer
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**inputs).waveform
        output_np = output.squeeze().cpu().numpy()
        scipy.io.wavfile.write(audio_file_path, rate=model.config.sampling_rate, data=output_np)
        return {"status": "Success", "audio_file": audio_file_path}
    except Exception as e:
        logger.error(f"Audio generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")