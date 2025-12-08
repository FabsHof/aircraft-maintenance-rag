import logging
import os

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.FileHandler('logs/app.log'), logging.StreamHandler()]
    )

def log_info(message):
    logging.info(message)