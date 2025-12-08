import os
from os import path
import glob
from src.util.logging import log_info

def process_data(raw_dir: str, clean_dir: str) -> None:
    '''
    Process raw data files from raw_dir and save cleaned files to clean_dir.
    arguments:
        raw_dir: Directory containing raw data files.
        clean_dir: Directory to save cleaned data files.
    returns:
        None
    '''
    os.makedirs(clean_dir, exist_ok=True)
    raw_files = glob.glob(path.join(raw_dir, '*.pdf'))
    for raw_file in raw_files:
        filename = path.basename(raw_file)
        clean_file = path.join(clean_dir, filename)
        # TODO: Implement actual data processing logic here: e.g., like text extraction
        # For now, just copy the raw file to clean file as a placeholder
        with open(raw_file, 'rb') as rf, open(clean_file, 'wb') as cf:
            cf.write(rf.read())
        log_info(f'Processing {raw_file} and saving to {clean_file}')

def main():
    raw_dir = path.join('data', 'raw')
    clean_dir = path.join('data', 'clean')
    if not path.exists(raw_dir):
        log_info(f'Raw data directory {raw_dir} does not exist. Please run the data loading step first.')
        return
    process_data(raw_dir, clean_dir)

if __name__ == '__main__':
    main()