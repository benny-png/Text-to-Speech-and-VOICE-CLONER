from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QStackedWidget, QProgressBar, QTextEdit, QScrollArea)
from PyQt6.QtCore import Qt
from helpers import StyledButton, StyledLineEdit, StyledComboBox
from threads import SetupThread, FinetuneThread, GenerateAudioThread

class MainWindow(QMainWindow):
    """
    MainWindow class for the Voice Cloner application.
    """
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
        setup_page = self.create_setup_page()
        self.stacked_widget.addWidget(setup_page)

        # Fine-tuning page
        finetune_page = self.create_finetune_page()
        self.stacked_widget.addWidget(finetune_page)

        # Audio generation page
        generate_page = self.create_generate_page()
        self.stacked_widget.addWidget(generate_page)

        # Documentation page
        documentation_page = self.create_documentation_page()
        self.stacked_widget.addWidget(documentation_page)

        # Navigation buttons
        nav_layout = QHBoxLayout()
        setup_nav = StyledButton("Setup", "#34495e")
        setup_nav.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        finetune_nav = StyledButton("Fine-tune", "#34495e")
        finetune_nav.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        generate_nav = StyledButton("Generate", "#34495e")
        generate_nav.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))
        documentation_nav = StyledButton("Documentation", "#34495e")
        documentation_nav.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(3))

        nav_layout.addWidget(setup_nav)
        nav_layout.addWidget(finetune_nav)
        nav_layout.addWidget(generate_nav)
        nav_layout.addWidget(documentation_nav)

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

    def create_setup_page(self):
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
        
        return setup_page

    def create_finetune_page(self):
        finetune_page = QWidget()
        finetune_layout = QVBoxLayout()
        finetune_page.setLayout(finetune_layout)

        finetune_label = QLabel("Fine-tuning")
        finetune_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        finetune_layout.addWidget(finetune_label)

        self.hf_dataset_repo = StyledLineEdit("Enter Hugging Face dataset repo (optional, e.g., 'username/repo')")
        finetune_layout.addWidget(self.hf_dataset_repo)

        self.dataset_dir = StyledLineEdit("Enter hugging Face dataset directory path")
        finetune_layout.addWidget(self.dataset_dir)

        self.csv_file = StyledLineEdit("Enter path to CSV file containing metadata")
        finetune_layout.addWidget(self.csv_file)

        self.model_type = StyledComboBox()
        self.model_type.addItems(["mms", "vits"])
        self.model_type.setPlaceholderText("Select model type (mms or vits)")  # Using a combobox, but no placeholder in Qt for real
        finetune_layout.addWidget(self.model_type)

        self.voice_name = StyledLineEdit("Enter a name for the voice you're creating")
        finetune_layout.addWidget(self.voice_name)

        self.language = StyledLineEdit("Enter language code (e.g., 'en' for English, 'sw' for Swahili)")
        finetune_layout.addWidget(self.language)

        finetune_button = StyledButton("Fine-tune", "#2ecc71")
        finetune_button.clicked.connect(self.finetune)
        finetune_layout.addWidget(finetune_button)

        finetune_layout.addStretch()

        return finetune_page

    def create_generate_page(self):
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
                color: #333;
            }
            QTextEdit::placeholder {
                color: #555;  /* Darker color for better contrast */
            }
        """)
        generate_layout.addWidget(self.text_input)

        self.filename_input = StyledLineEdit("Enter desired output filename (e.g., 'output.wav')")
        generate_layout.addWidget(self.filename_input)

        self.model_name_input = StyledLineEdit("Enter the name of the hugginface model to use. (e.g. , 'Benjamin-png/swahili-mms-tts-finetuned', 'facebook/mms-tts-eng')")
        generate_layout.addWidget(self.model_name_input)

        generate_button = StyledButton("Generate Audio", "#e74c3c")
        generate_button.clicked.connect(self.generate_audio)
        generate_layout.addWidget(generate_button)

        generate_layout.addStretch()

        return generate_page

    def create_documentation_page(self):
        documentation_page = QWidget()
        documentation_layout = QVBoxLayout()
        documentation_page.setLayout(documentation_layout)

        documentation_label = QLabel("Documentation")
        documentation_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        documentation_layout.addWidget(documentation_label)

        # Scroll Area for Documentation
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        documentation_layout.addWidget(scroll_area)

        # Container Widget for Scroll Area
        container = QWidget()
        scroll_area.setWidget(container)
        container_layout = QVBoxLayout()
        container.setLayout(container_layout)

        documentation_text = QTextEdit()
        documentation_text.setReadOnly(True)
        documentation_text.setStyleSheet("""
            QTextEdit {
                border: none;
                background-color: #f0f0f0;
                color: #333;
                font-size: 14px;
                line-height: 1.5;
                padding: 10px;
            }
            h1 {
                color: #3498db;
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 10px;
            }
            h2 {
                color: #2ecc71;
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 10px;
            }
            p {
                margin-bottom: 10px;
                text-align: justify;
            }
            ul {
                margin-bottom: 10px;
            }
            li {
                margin-bottom: 5px;
            }
        """)
        documentation_text.setHtml("""
        <h1>Welcome to the Voice Cloner Documentation!</h1>

        <h2>Setup Page</h2>
        <ul>
            <li><b>Hugging Face Token:</b> Enter your Hugging Face token for authentication.</li>
            <li>Click the <b>Setup</b> button to initialize the setup process.</li>
        </ul>

        <h2>Fine-tuning Page</h2>
        <ul>
            <li><b>APPROACH 1</b> You already have the structured dataset in Huggingface</li>
            <li><b>Hugging Face Dataset Repository:</b> Enter the Hugging Face dataset repository (optional).</li>
            <li><b>Local Dataset Directory:</b> NOT APPLICABLE HERE. Leave empty</li>
            <li><b>CSV File:</b> NOT APPLICABLE HERE. Leave empty</li>
            <li><b>Model Type:</b> Select the model type ('mms' or 'vits').</li>
            <li><b>Voice Name:</b> Enter a name for the voice you're creating.</li>
            <li><b>Language Code:</b> Enter the language code (e.g., 'en' for English, 'sw' for Swahili).</li>
            <li>Click the <b>Fine-tune</b> button to start fine-tuning.</li>
        </ul>
        

        <ul>
            <li><b>APPROACH 2</b> You are preparing the dataset Locally. This approach handles pushing the data to hugginface </li>
            <li><b>Hugging Face Dataset Repository:</b>NOT APPLICABLE HERE. Leave empty.</li>
            <li><b>Local Dataset Directory:</b> Enter Path to the directory containing .wav  the Audio Files. NB in wav format in name pattern "001.wav,002.wav,...".</li>
            <li><b>CSV File:</b> Enter the path to the CSV file containing metadata. The CSV file should have the following columns:
                * file_name: Name of the .wav file (e.g., "001.wav")
                * text: Corresponding text for the audio file
                * speaker_id: ID of the speaker (optional, default is 1)
                Example CSV content:
                    file_name,text,speaker_id
                    001.wav,This is the first sentence.,1
                    002.wav,This is the second sentence.,1</li>
            <li><b>Model Type:</b> Select the model type ('mms' or 'vits').</li>
            <li><b>Voice Name:</b> Enter a name for the voice you're creating.</li>
            <li><b>Language Code:</b> Enter the language code (e.g., 'en' for English, 'sw' for Swahili).</li>
            <li>Click the <b>Fine-tune</b> button to start fine-tuning.</li>
        </ul>

        <h2>Audio Generation Page</h2>
        <ul>
            <li><b>Text Input:</b> Enter the text you want to convert to speech.</li>
            <li><b>Output Filename:</b> Enter the desired output filename (e.g., 'output.wav').</li>
            <li><b>Model Name:</b> Enter the name of the hugginface model to use. (e.g. , 'Benjamin-png/swahili-mms-tts-finetuned', 'facebook/mms-tts-eng') </li>
            <li>Click the <b>Generate Audio</b> button to generate audio.</li>
        </ul>

        <p>For more information, visit our website or contact support.</p>
        """)
        container_layout.addWidget(documentation_text)
        
        return documentation_page

    def setup(self):
        """
        Perform setup by invoking the setup endpoint of the FastAPI server.
        """
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
        """
        Handle the completion of the setup process.
        """
        if success:
            self.status_label.setText(message)
            self.progress_bar.setValue(100)
        else:
            self.status_label.setText(f"Setup failed: {message}")
            self.progress_bar.setValue(0)

    def finetune(self):
        """
        Perform fine-tuning by invoking the finetune endpoint of the FastAPI server.
        """
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
        """
        Handle the completion of the fine-tuning process.
        """
        if success:
            self.status_label.setText(message)
            self.progress_bar.setValue(100)
        else:
            self.status_label.setText(f"Fine-tuning failed: {message}")
            self.progress_bar.setValue(0)

    def generate_audio(self):
        """
        Perform audio generation by invoking the generate_audio endpoint of the FastAPI server.
        """
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
        """
        Handle the completion of the audio generation process.
        """
        if success:
            self.status_label.setText(message)
            self.progress_bar.setValue(100)
        else:
            self.status_label.setText(f"Audio generation failed: {message}")
            self.progress_bar.setValue(0)