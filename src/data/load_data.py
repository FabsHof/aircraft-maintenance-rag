import requests
from bs4 import BeautifulSoup
import os
from src.util.logging import log_message

def load_file(url: str, file_path: str) -> None:
    '''Download a file from a URL to a specified file path.'''
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad status codes
    with open(file_path, 'wb') as file:
        file.write(response.content)

def download_files_from_links(selection_fn: callable, url: str, max_files: int = None, raw_dir: str = None) -> None:
    '''
    download files from links selected by selection_fn on the given URL page.
    arguments:
        selection_fn: A function that takes a BeautifulSoup object and returns a list of link elements.
        url: The URL of the webpage to scrape links from
        max_files: Maximum number of files to download. If None, download all.
        raw_dir: Directory to save downloaded files. If None, defaults to 'data/raw'.
    returns:
        None
    '''
    if raw_dir is None:
        raw_dir = os.path.join('data', 'raw')
    os.makedirs(raw_dir, exist_ok=True)

    # Fetch the webpage
    response = requests.get(url)
    response.raise_for_status()
    
    # Parse HTML
    soup = BeautifulSoup(response.content, 'html.parser')
    links = selection_fn(soup)
    
    if not links:
        log_message('No links found on the page.')
        return
    
    # Filter for PDF files only
    links = [link for link in links if link['href'].lower().endswith('.pdf')]
    if max_files:
        links = links[:max_files]
    for link in links:
        href = link['href']
        # Make absolute URL if relative
        if not href.startswith('http'):
            href = f'https://www.faa.gov{href}'
        
        # Extract filename from URL
        filename = os.path.join(raw_dir, href.split('/')[-1])
        try:
            log_message(f'Downloading {filename}...')
            load_file(href, filename)
            log_message(f'Successfully downloaded {filename}')
        except Exception as e:
            log_message(f'Failed to download {href}: {e}')
def main():
    def selection_fn(soup):
        '''Select links from the table with caption "Reference Handbooks".'''
        target_table = None
        tables = soup.find_all('table')
        for table in tables:
            caption = table.find('caption')
            if caption and 'Reference Handbooks' in caption.get_text():
                target_table = table
                break
        
        if not target_table:
            log_message('Table with "Reference Handbooks" caption not found')
            return
        return target_table.find_all('a', href=True)

    url = 'https://www.faa.gov/regulations_policies/handbooks_manuals/aviation'
    download_files_from_links(selection_fn, url)

if __name__ == '__main__':
    main()
    