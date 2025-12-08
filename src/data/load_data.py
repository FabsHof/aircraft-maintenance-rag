import json
import argparse
import requests
import os
from os import path
from src.util.logging import log_info

def load_file(url: str, file_path: str) -> None:
    '''Download a file from a URL to a specified file path.'''
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for bad status codes
    with open(file_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

def main(config_path: str) -> None:
    '''
    Download documents specified in the configuration file. The config file should be a JSON
    with a list of 'documents', each containing at least 'url' and 'id' fields.
    arguments:
        config_path: Path to the configuration JSON file.
    returns:
        None
    '''

    if not path.exists(config_path):
        log_info(f'⚡️ Configuration file {config_path} does not exist.')
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    if not config.get('documents'):
        log_info('⚡️ No documents found in the configuration file.')
        return
    project = config.get('project', 'default_project')
    version = config.get('version', '1.0')
    base_dir = path.join(config.get('base_directory', 'data'), project, version)
    
    for i, doc in enumerate(config['documents']):
            url = doc.get('url')
            id = doc.get('id')
            if not url or not id:
                log_info(f'⚡️ No URL or ID found for document at index {i}. Skipping download.')
                continue
            
            title = doc.get('title', f'document_{id}')
            if doc.get('direct_download', True):
                # Create directories which are specific to the type of document
                directory = doc.get('directory', '')
                current_dir = path.join(base_dir, directory)
                os.makedirs(current_dir, exist_ok=True)
                
                filename = url.split('/')[-1]
                file_path = path.join(current_dir, filename)
                try:
                    log_info(f'⏳ Downloading {filename} from {url}...')
                    load_file(url, file_path)
                    log_info(f'✅ Successfully downloaded {filename}')
                except Exception as e:
                    log_info(f'⚡️ Failed to download {url}:\n{e}')
            else:
                log_info(f'⚠️ Document ID {id} "{title}" does not support direct download!\nPlease, download manually and place it in directory "{directory}".\nURL: {url}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download aircraft-related documents from the config file. Defaults to Airbus A320-related and generic FAA documents.')
    parser.add_argument('--config', '-c', type=str, default=path.join('config', 'airbus_a320.json'), help='Path to the configuration file specifying document sources')
    args = parser.parse_args()

    main(args.config)
    