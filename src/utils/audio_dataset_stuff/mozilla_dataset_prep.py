import os
from typing import List, Dict
from datasets import load_dataset, Dataset, Audio
from huggingface_hub import HfApi, HfFolder, create_repo

def create_and_upload_swahili_dataset(repo_id: str, client_id: str) -> None:
    """
    Create a dataset from the Mozilla Common Voice Swahili dataset for a specific client_id and upload it to Hugging Face.

    Args:
    repo_id (str): Hugging Face repository ID for uploading the dataset
    client_id (str): The specific client_id to filter the dataset
    """

    # Load the Swahili dataset
    original_dataset = load_dataset("mozilla-foundation/common_voice_17_0", "sw")

    # Filter the dataset for the specific client_id
    filtered_dataset = original_dataset.filter(lambda example: example['client_id'] == client_id)

    # Create dataset entries
    data = create_dataset_entries(filtered_dataset["train"])

    # Create Dataset
    dataset = Dataset.from_list(data)

    # Add audio feature to the dataset
    dataset = dataset.cast_column("audio", Audio())

    # Upload to Hugging Face
    upload_to_huggingface(dataset, repo_id)

def create_dataset_entries(filtered_dataset) -> List[Dict]:
    """Create dataset entries based on the filtered dataset."""
    data = []
    for i, item in enumerate(filtered_dataset):
        entry = {
            'line_id': f"SW{i:04d}",
            'audio': item['audio']['array'],
            'text': item['sentence'],
            'speaker_id': item['client_id']
        }
        data.append(entry)
    return data

def upload_to_huggingface(dataset: Dataset, repo_id: str) -> None:
    """Upload the dataset to Hugging Face."""
    hf_token = HfFolder.get_token()
    api = HfApi()

    try:
        create_repo(repo_id=repo_id, repo_type="dataset", token=hf_token)
        print("Repository created successfully.")
    except Exception as e:
        print(f"Repository creation failed or already exists: {e}")

    dataset.push_to_hub(repo_id, token=hf_token)
    print("Dataset uploaded successfully!")

# Example usage:
specific_client_id = "052c5091df7681302a2117b2d21db1540c2156f5254ebe9876a7d0146588eab582e11cb47761a18f84200a510a5386bdf024374f76113cd15fe1cc8d7b9fcf0b"
create_and_upload_swahili_dataset("swahili-common-voice-woman_sound", specific_client_id)