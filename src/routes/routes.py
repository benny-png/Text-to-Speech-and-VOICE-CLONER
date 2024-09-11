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
    voice_name: str = Form(...),
    language: str = Form(default="sw")
):
    """
    Fine-tune a VITS or MMS model based on the provided dataset, model type, voice name, and language (Swahili as default).
    
    Args:
        dataset: Uploaded dataset for fine-tuning.
        model_type: 'vits' or 'mms'.
        voice_name: Custom voice name for the fine-tuning model.
        language: Language code for MMS model, default 'sw' for Swahili.
    """
    try:
        # Step 1: Save uploaded dataset in src/files/datasets
        dataset_folder = Path(f"src/files/datasets/{voice_name}")
        dataset_folder.mkdir(parents=True, exist_ok=True)
        dataset_path = dataset_folder / "dataset.zip"
        
        with open(dataset_path, "wb") as f:
            shutil.copyfileobj(dataset.file, f)
        
        # Step 2: Check Hugging Face CLI is installed and logged in
        if not check_huggingface_cli_installed():
            install_huggingface_cli()
        
        # Ensure the user is logged in
        token = None
        with open(HF_TOKEN_PATH, 'r') as token_file:
            token = token_file.read().strip()
        
        if not token:
            return {"status": "Error", "message": "No Hugging Face token found. Please log in first."}
        
        login_huggingface(token)

        # Step 3: Set model repository based on model type, voice name, and language
        hub_model_id = f"Benjamin-png/{model_type}-tts-{language}-{voice_name}-finetuned"
        
        # Step 4: Create repository if it doesn't exist
        subprocess.run(["huggingface-cli", "repo", "create", hub_model_id, "--token", token], check=True)
        
        # Step 5: Start the fine-tuning process
        config = {
            "project_name": f"{voice_name}_vocals",
            "push_to_hub": True,
            "hub_model_id": hub_model_id,
            "overwrite_output_dir": True,
            "output_dir": f"src/files/outputs/{voice_name}_tts_finetuned",

            "dataset_name": str(dataset_path),
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

            # Use Swahili checkpoint for model_name_or_path
            "model_name_or_path": "Benjamin-png/swa-checkpoint",

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

            "fp16": True,
            "seed": 456
        }

        # Save configuration file dynamically for each voice
        config_folder = Path(f"src/files/config/{voice_name}")
        config_folder.mkdir(parents=True, exist_ok=True)
        config_path = config_folder / f"{voice_name}_config.json"
        
        with open(config_path, "w") as config_file:
            json.dump(config, config_file, indent=4)

        # Step 6: Run the fine-tuning process using the config
        subprocess.run(
            ["accelerate", "launch", "src/utils/finetune-hf-vits/run_vits_finetuning.py", str(config_path)],
            check=True
        )

        return {"status": "Success", "message": f"Model fine-tuned successfully and uploaded to {hub_model_id}"}

    except subprocess.CalledProcessError as e:
        return {"status": "Error", "message": f"An error occurred: {str(e)}"}



class AudioGenerationRequest(BaseModel):
    text: str
    filename: str
    model_name: str

def sanitize_filename(filename: str) -> str:
    """
    Sanitize the filename to ensure it is a valid path.

    Args:
        filename: The desired filename.

    Returns:
        A sanitized filename.
    """
    sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)  # Remove invalid characters
    return sanitized[:50]  # Limit filename length to 50 characters

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

    # Sanitize the filename
    sanitized_filename = sanitize_filename(filename)
    audio_file_path = f"src/files/outputs/{sanitized_filename}_tts.wav"

    try:
        # Load model and tokenizer dynamically based on the provided model name
        model = VitsModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Step 1: Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt")

        # Step 2: Generate waveform
        with torch.no_grad():
            output = model(**inputs).waveform

        # Step 3: Convert PyTorch tensor to NumPy array
        output_np = output.squeeze().cpu().numpy()

        # Step 4: Write to WAV file
        scipy.io.wavfile.write(audio_file_path, rate=model.config.sampling_rate, data=output_np)

        # Step 5: Return the audio file path
        return {"status": "Success", "audio_file": audio_file_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")
