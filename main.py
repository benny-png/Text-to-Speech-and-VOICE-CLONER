from src.pyqt_api_based.main import MainWindow
from PyQt6.QtWidgets import QApplication
import sys


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()  # Instantiate your MainWindow
    window.show()  # Show the window
    sys.exit(app.exec())  # Start the event loop and exit cleanly