import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)

def get_current_timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def log_message(message):
    timestamp = get_current_timestamp()
    logging.info(f'[{timestamp}] {message}')