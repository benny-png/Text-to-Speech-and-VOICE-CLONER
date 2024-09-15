# Voice Cloner

Voice Cloner is a Python application that enables users to fine-tune Text-to-Speech (TTS) models and generate audio from text. It features a user-friendly GUI built with PyQt6 and a backend powered by FastAPI. 

![PYQT UI](https://github.com/benny-png/VOICE-CLONER-FASTAPI/blob/master/images/pyqt_ui.png)

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Documentation](#documentation)
- [License](#license)
- [Contributing](#contributing)
## Supported languages
For fine tuning Supported languages are (e.g., 'en' for English, 'sw' for Swahili, 'ko' for Korean, 'mr' for Marathi, 'ta' for Tamil, 'gu' for Gujarati). 

For generating audio you can use your fine tuned hugginface model (just put your repo) or you can use existing models of different languages by facebook [here](https://huggingface.co/models?search=facebook/mms-tts)

![PYQT UI](https://github.com/benny-png/VOICE-CLONER-FASTAPI/blob/master/images/generate_audio.png)


## Features
- **Setup**: Easily set up the environment with necessary dependencies and Hugging Face authentication.
- **Fine-tuning**: Supports fine-tuning of MMS and VITS models using Hugging Face datasets or local datasets.
- **Audio Generation**: Generate audio from text using fine-tuned models.
- **Documentation**: Built-in documentation for easy reference.

## Requirements
- Python 3.8+
- PyQt6
- FastAPI
- Hugging Face Transformers
- scipy
- psutil

## Installation

### Clone the Repository:
```bash
git clone https://github.com/benny-png/VOICE-CLONER-FASTAPI.git
cd VOICE-CLONER-FASTAPI
```

### Create and Activate a Virtual Environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Start the Application
Run the main script:
```bash
python main.py
```
This will start the PyQt6 application. The FastAPI server is assumed to be running separately.

### Setup
1. **Enter Hugging Face Token**: On the setup page, enter your Hugging Face token.
2. **Click Setup**: Click the 'Setup' button to install dependencies and log in to Hugging Face.

### Fine-tuning

![PYQT UI](https://github.com/benny-png/VOICE-CLONER-FASTAPI/blob/master/images/finetune.png)


#### Approach 1: Using an Existing Structured Dataset on Hugging Face
1. **Hugging Face Dataset Repository**: Enter the repository name (optional).
2. **Model Type**: Select the model type ('mms' or 'vits').
3. **Voice Name**: Enter a name for the custom voice model.
4. **Language Code**: Enter the language code (e.g., 'en', 'sw').
5. **Click Fine-tune**: Begin the fine-tuning process.

#### Approach 2: Preparing and Pushing the Dataset Locally
1. **Local Dataset Directory**: Provide the path to the .wav audio files.
2. **CSV File**: Provide a CSV file with columns:
   - `file_name`: The .wav file name (e.g., "001.wav").
   - `text`: The corresponding text.
   - `speaker_id`: Speaker ID (optional).
3. **Model Type**: Select the model type ('mms' or 'vits').
4. **Voice Name**: Enter the name of the custom voice model.
5. **Click Fine-tune**: Start fine-tuning.

### Audio Generation
1. **Enter Text**: On the 'Audio Generation' page, input the text to convert to speech.
2. **Enter Output Filename**: Provide an output filename for the audio.
3. **Enter Model Name**: Specify the Hugging Face model name.
4. **Click Generate Audio**: Generate the audio file.

### Stop the Application
To stop the application, simply close the GUI window.

## File Structure
- `main.py`: Main script to start the application.
- `src\pyqt_api_based\ui.py`: PyQt6 UI code for the application.
- `src\pyqt_api_based\threads.py`: Handles setup, fine-tuning, and audio generation tasks.
- `src\routes\routes.py`: FastAPI routes for backend operations.
- `src\pyqt_api_based\helpers.py`: Custom widgets for PyQt6 UI.
- `requirements.txt`: List of dependencies.

## Documentation

![PYQT UI](https://github.com/benny-png/VOICE-CLONER-FASTAPI/blob/master/images/documentation.png)


### Setup Page
- **Hugging Face Token**: Enter your Hugging Face token.
- Click the **Setup** button to initialize.

### Fine-tuning Page


#### Approach 1: Using Hugging Face Dataset
- Enter Hugging Face Dataset Repository (optional).
- Select the **Model Type** ('mms' or 'vits').
- Enter the **Voice Name** and **Language Code**.
- Click **Fine-tune**.

#### Approach 2: Using Local Dataset
- Enter the **Local Dataset Directory** and **CSV File**.
- Select the **Model Type**.
- Enter the **Voice Name** and **Language Code**.
- Click **Fine-tune**.

### Audio Generation Page
- Enter text, the output filename, and the Hugging Face model name.
- Click **Generate Audio**.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for enhancements or bug fixes.
