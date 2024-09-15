import sys
import subprocess
from PyQt6.QtWidgets import QApplication
from ui import MainWindow
from logging_config import setup_logging

def start_server():
    """
    Start the FastAPI server using uvicorn as a subprocess.
    
    Returns:
        subprocess.Popen: The subprocess running the uvicorn server.
    """
    return subprocess.Popen(["uvicorn", "src.routes.routes:app"])

if __name__ == "__main__":
    setup_logging()

    #server_process = start_server()
    
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
    #finally:
        #server_process.terminate()