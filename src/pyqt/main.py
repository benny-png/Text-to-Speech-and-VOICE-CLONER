import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLineEdit, QTextEdit, 
                             QLabel, QFileDialog, QComboBox, QProgressBar,
                             QStackedWidget, QScrollArea)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QIcon
import requests

class StyledButton(QPushButton):
    def __init__(self, text, color):
        super().__init__(text)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {color}99;
            }}
        """)

class StyledLineEdit(QLineEdit):
    def __init__(self, placeholder):
        super().__init__()
        self.setPlaceholderText(placeholder)
        self.setStyleSheet("""
            QLineEdit {
                border: 2px solid #ddd;
                border-radius: 5px;
                padding: 5px;
                background-color: #f5f5f5;
            }
            QLineEdit:focus {
                border-color: #3498db;
            }
            QLineEdit::placeholder {
                color: #999;
            }
        """)

class StyledComboBox(QComboBox):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QComboBox {
                border: 2px solid #ddd;
                border-radius: 5px;
                padding: 5px;
                background-color: #f5f5f5;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 30px;
                border-left-width: 1px;
                border-left-color: #ddd;
                border-left-style: solid;
                border-top-right-radius: 5px;
                border-bottom-right-radius: 5px;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
            }
        """)


class SetupThread(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, token):
        super().__init__()
        self.token = token

    def run(self):
        try:
            response = requests.post("http://localhost:8000/setup", data={"token": self.token})
            if response.status_code == 200:
                self.finished.emit(True, "Setup completed successfully.")
            else:
                self.finished.emit(False, f"Setup failed: {response.json()['detail']}")
        except Exception as e:
            self.finished.emit(False, f"Error during setup: {str(e)}")

class FinetuneThread(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            response = requests.post("http://localhost:8000/finetune", data=self.params)
            if response.status_code == 200:
                self.finished.emit(True, "Fine-tuning completed successfully.")
            else:
                self.finished.emit(False, f"Fine-tuning failed: {response.json()['detail']}")
        except Exception as e:
            self.finished.emit(False, f"Error during fine-tuning: {str(e)}")

class GenerateAudioThread(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            response = requests.post("http://localhost:8000/generate_audio", json=self.params)
            if response.status_code == 200:
                self.finished.emit(True, f"Audio generated: {response.json()['audio_file']}")
            else:
                self.finished.emit(False, f"Audio generation failed: {response.json()['detail']}")
        except Exception as e:
            self.finished.emit(False, f"Error during audio generation: {str(e)}")



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Cloner")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QLabel {
                font-size: 14px;
                color: #333;
            }
            QTextEdit {
                border: 2px solid #ddd;
                border-radius: 5px;
                padding: 5px;
                background-color: #fff;
            }
            QTextEdit::placeholder {
                color: #999;
            }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # Title
        title_label = QLabel("Voice Cloner")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
            margin: 20px 0;
        """)
        main_layout.addWidget(title_label)

        # Stacked widget for different sections
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)

        # Setup page
        setup_page = QWidget()
        setup_layout = QVBoxLayout()
        setup_page.setLayout(setup_layout)

        setup_label = QLabel("Setup")
        setup_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        setup_layout.addWidget(setup_label)

        self.token_input = StyledLineEdit("Enter your Hugging Face token for authentication")
        setup_layout.addWidget(self.token_input)

        setup_button = StyledButton("Setup", "#3498db")
        setup_button.clicked.connect(self.setup)
        setup_layout.addWidget(setup_button)

        setup_layout.addStretch()

        # Fine-tuning page
        finetune_page = QWidget()
        finetune_layout = QVBoxLayout()
        finetune_page.setLayout(finetune_layout)

        finetune_label = QLabel("Fine-tuning")
        finetune_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        finetune_layout.addWidget(finetune_label)

        self.hf_dataset_repo = StyledLineEdit("Enter Hugging Face dataset repo (optional, e.g., 'username/repo')")
        finetune_layout.addWidget(self.hf_dataset_repo)

        self.dataset_dir = StyledLineEdit("Enter local dataset directory path")
        finetune_layout.addWidget(self.dataset_dir)

        self.csv_file = StyledLineEdit("Enter path to CSV file containing metadata")
        finetune_layout.addWidget(self.csv_file)

        self.model_type = StyledComboBox()
        self.model_type.addItems(["mms", "vits"])
        self.model_type.setPlaceholderText("Select model type (mms or vits)")
        finetune_layout.addWidget(self.model_type)

        self.voice_name = StyledLineEdit("Enter a name for the voice you're creating")
        finetune_layout.addWidget(self.voice_name)

        self.language = StyledLineEdit("Enter language code (e.g., 'en' for English, 'sw' for Swahili)")
        finetune_layout.addWidget(self.language)

        finetune_button = StyledButton("Fine-tune", "#2ecc71")
        finetune_button.clicked.connect(self.finetune)
        finetune_layout.addWidget(finetune_button)

        finetune_layout.addStretch()

        # Audio generation page
        generate_page = QWidget()
        generate_layout = QVBoxLayout()
        generate_page.setLayout(generate_layout)

        generate_label = QLabel("Audio Generation")
        generate_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        generate_layout.addWidget(generate_label)

        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter the text you want to convert to speech")
        self.text_input.setStyleSheet("""
            QTextEdit {
                border: 2px solid #ddd;
                border-radius: 5px;
                padding: 5px;
                background-color: #fff;
            }
            QTextEdit::placeholder {
                color: #999;
            }
        """)
        
        
        generate_layout.addWidget(self.text_input)

        self.filename_input = StyledLineEdit("Enter desired output filename (e.g., 'output.wav')")
        generate_layout.addWidget(self.filename_input)

        self.model_name_input = StyledLineEdit("Enter the name of the fine-tuned model to use")
        generate_layout.addWidget(self.model_name_input)

        generate_button = StyledButton("Generate Audio", "#e74c3c")
        generate_button.clicked.connect(self.generate_audio)
        generate_layout.addWidget(generate_button)

        generate_layout.addStretch()

        # Add pages to stacked widget
        self.stacked_widget.addWidget(setup_page)
        self.stacked_widget.addWidget(finetune_page)
        self.stacked_widget.addWidget(generate_page)

        # Navigation buttons
        nav_layout = QHBoxLayout()
        setup_nav = StyledButton("Setup", "#34495e")
        setup_nav.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        finetune_nav = StyledButton("Fine-tune", "#34495e")
        finetune_nav.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        generate_nav = StyledButton("Generate", "#34495e")
        generate_nav.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))

        nav_layout.addWidget(setup_nav)
        nav_layout.addWidget(finetune_nav)
        nav_layout.addWidget(generate_nav)

        main_layout.addLayout(nav_layout)

        # Status display
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            font-size: 14px;
            color: #333;
            margin-top: 10px;
        """)
        main_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #ddd;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                width: 10px;
                margin: 0.5px;
            }
        """)
        main_layout.addWidget(self.progress_bar)

    def setup(self):
        token = self.token_input.text()
        if not token:
            self.status_label.setText("Please enter a Hugging Face token")
            return

        self.status_label.setText("Setting up...")
        self.progress_bar.setValue(0)

        self.setup_thread = SetupThread(token)
        self.setup_thread.finished.connect(self.setup_finished)
        self.setup_thread.start()

    def setup_finished(self, success, message):
        if success:
            self.status_label.setText(message)
            self.progress_bar.setValue(100)
        else:
            self.status_label.setText(f"Setup failed: {message}")
            self.progress_bar.setValue(0)

    def finetune(self):
        params = {
            "hf_dataset_repo": self.hf_dataset_repo.text(),
            "dataset_dir": self.dataset_dir.text(),
            "csv_file": self.csv_file.text(),
            "model_type": self.model_type.currentText(),
            "voice_name": self.voice_name.text(),
            "language": self.language.text()
        }

        self.status_label.setText("Fine-tuning in progress...")
        self.progress_bar.setValue(0)

        self.finetune_thread = FinetuneThread(params)
        self.finetune_thread.finished.connect(self.finetune_finished)
        self.finetune_thread.start()

    def finetune_finished(self, success, message):
        if success:
            self.status_label.setText(message)
            self.progress_bar.setValue(100)
        else:
            self.status_label.setText(f"Fine-tuning failed: {message}")
            self.progress_bar.setValue(0)

    def generate_audio(self):
        params = {
            "text": self.text_input.toPlainText(),
            "filename": self.filename_input.text(),
            "model_name": self.model_name_input.text()
        }

        self.status_label.setText("Generating audio...")
        self.progress_bar.setValue(0)

        self.generate_thread = GenerateAudioThread(params)
        self.generate_thread.finished.connect(self.generate_finished)
        self.generate_thread.start()

    def generate_finished(self, success, message):
        if success:
            self.status_label.setText(message)
            self.progress_bar.setValue(100)
        else:
            self.status_label.setText(f"Audio generation failed: {message}")
            self.progress_bar.setValue(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())