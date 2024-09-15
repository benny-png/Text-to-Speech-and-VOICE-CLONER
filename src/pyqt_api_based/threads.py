from PyQt6.QtCore import QThread, pyqtSignal
import requests

class SetupThread(QThread):
    """
    QThread to handle setup operations for setting up the environment, such as installing dependencies and logging in to Hugging Face.
    """
    finished = pyqtSignal(bool, str)

    def __init__(self, token):
        super().__init__()
        self.token = token

    def run(self):
        """
        Execute the setup process in a separate thread.
        """
        try:
            response = requests.post("http://localhost:8000/setup", data={"token": self.token})
            if response.status_code == 200:
                self.finished.emit(True, "Setup completed successfully.")
            else:
                self.finished.emit(False, f"Setup failed: {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            self.finished.emit(False, f"Error during setup: {str(e)}")

class FinetuneThread(QThread):
    """
    QThread to handle the fine-tuning process of models.
    """
    finished = pyqtSignal(bool, str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        """
        Execute the fine-tuning process in a separate thread.
        """
        try:
            response = requests.post("http://localhost:8000/finetune", data=self.params)
            if response.status_code == 200:
                self.finished.emit(True, "Fine-tuning completed successfully.")
            else:
                self.finished.emit(False, f"Fine-tuning failed: {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            self.finished.emit(False, f"Error during fine-tuning: {str(e)}")

class GenerateAudioThread(QThread):
    """
    QThread to handle the audio generation process.
    """
    finished = pyqtSignal(bool, str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        """
        Execute the audio generation process in a separate thread.
        """
        try:
            response = requests.post("http://localhost:8000/generate_audio", json=self.params)
            if response.status_code == 200:
                self.finished.emit(True, f"Audio generated: {response.json()['audio_file']}")
            else:
                self.finished.emit(False, f"Audio generation failed: {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            self.finished.emit(False, f"Error during audio generation: {str(e)}")