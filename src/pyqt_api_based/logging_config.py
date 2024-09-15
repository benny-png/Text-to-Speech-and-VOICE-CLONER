import logging

def setup_logging():
    logging.basicConfig(filename='src/pyqt_api_based/assets/voice_cloner.log', level=logging.DEBUG, 
                        format='%(asctime)s - %(levelname)s - %(message)s')