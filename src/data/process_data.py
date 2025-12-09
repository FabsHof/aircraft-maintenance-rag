import os
import json
from os import path
from src.util.logging import log_info
import argparse

def get_config_files(config_dir: str) -> list[str]:
    '''
    Get all JSON configuration files from the config directory.
    arguments:
        config_dir: Directory containing configuration files.
    returns: List of JSON configuration file paths.
    '''
    return [path.join(config_dir, f) for f in os.listdir(config_dir) if f.endswith('.json')]


def process_config_file(config_file: str, raw_dir: str, clean_dir: str) -> None:
    '''
    Process all raw files for a single configuration file.
    arguments:
        config_file: Path to the configuration file.
        raw_dir: Directory containing raw data.
        clean_dir: Directory to save clean data.
    returns: None
    '''
    log_info(f'Processing data for configuration file: {config_file}')
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    project = config.get('project', 'default_project')
    version = config.get('version', '1.0')
    project_raw_dir = path.join(raw_dir, project, version)
    project_clean_dir = path.join(clean_dir, project, version)
    
    if not path.exists(project_raw_dir):
        log_info(f'Raw directory {project_raw_dir} does not exist. Skipping this configuration.')
        return
    
    sub_dirs = [d for d in os.listdir(project_raw_dir) if path.isdir(path.join(project_raw_dir, d))]
    for sub_dir in sub_dirs:
        os.makedirs(path.join(project_clean_dir, sub_dir), exist_ok=True)

        project_raw_files = [path.join(project_raw_dir, sub_dir, f) for f in os.listdir(path.join(project_raw_dir, sub_dir))]
        if not project_raw_files:
            log_info(f'No raw files found in {path.join(project_raw_dir, sub_dir)}. Skipping this configuration.')
            return
        
        os.makedirs(path.join(project_clean_dir, sub_dir), exist_ok=True)
        log_info(f'Processing {len(project_raw_files)} files for project {project}, version {version}.')
        
        for raw_file in project_raw_files:
            process_raw_file(raw_file, path.join(project_clean_dir, sub_dir))
            
def process_raw_file(raw_file: str, clean_dir: str) -> None:
    '''
    Process a single raw file and save it to the clean directory.
    arguments:
        raw_file: Path to the raw file.
        clean_dir: Directory to save the clean file.
    returns: None
    '''
    with open(raw_file, 'rb') as rf:
        data = rf.read()
    # TODO: Implement actual data processing logic here
    filename = path.basename(raw_file)
    clean_file = path.join(clean_dir, filename)
    
    with open(clean_file, 'wb') as cf:
        cf.write(data)
    
    log_info(f'Processed {raw_file} and saved to {clean_file}')

def main(config_dir: str, raw_dir: str, clean_dir: str) -> None:

    if not path.exists(config_dir):
        log_info(f'Configuration directory {config_dir} does not exist. Please provide a valid directory.')
        return
    if not path.exists(raw_dir):
        log_info(f'Raw data directory {raw_dir} does not exist. Please run the data loading step first.')
        return
    os.makedirs(clean_dir, exist_ok=True)

    config_files = get_config_files(config_dir)
    if not config_files:
        log_info(f'No configuration files found in {config_dir}. Please provide at least one config file.')
        return
    
    for config_file in config_files:
        process_config_file(config_file, raw_dir, clean_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process raw data into clean data.')
    parser.add_argument('--config_dir', type=str, default='config', help='Directory containing configuration files')
    parser.add_argument('--raw_dir', type=str, default='data/raw', help='Directory containing raw data')
    parser.add_argument('--clean_dir', type=str, default='data/clean', help='Directory to save clean data')
    args = parser.parse_args()

    main(args.config_dir, args.raw_dir, args.clean_dir)