import requests
from bs4 import BeautifulSoup
import os

def load_file(url: str, file_path: str) -> None:
    '''Download a file from a URL to a specified file path.'''
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad status codes
    with open(file_path, 'wb') as file:
        file.write(response.content)

def main(max_files: int = None, url: str = None) -> None:
    # Fetch the webpage
    response = requests.get(url)
    response.raise_for_status()
    
    # Parse HTML
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the table with caption 'Reference Handbooks'
    target_table = None
    tables = soup.find_all('table')
    for table in tables:
        caption = table.find('caption')
        if caption and 'Reference Handbooks' in caption.get_text():
            target_table = table
            break
    
    if not target_table:
        print('Table with "Reference Handbooks" caption not found')
        return
    
    # Create data/manuals directory
    os.makedirs(os.path.join('data', 'manuals'), exist_ok=True)
    
    # Find all pdf links in the table
    links = target_table.find_all('a', href=True)
    links = [link for link in links if link['href'].lower().endswith('.pdf')]
    if max_files:
        links = links[:max_files]
    for link in links:
        href = link['href']
        # Make absolute URL if relative
        if not href.startswith('http'):
            href = f'https://www.faa.gov{href}'
        
        # Extract filename from URL
        filename = os.path.join('data', 'manuals', href.split('/')[-1])
        try:
            print(f'Downloading {filename}...')
            load_file(href, filename)
            print(f'Successfully downloaded {filename}')
        except Exception as e:
            print(f'Failed to download {href}: {e}')
    url = 'https://www.faa.gov/regulations_policies/handbooks_manuals/aviation'

if __name__ == '__main__':
    url = 'https://www.faa.gov/regulations_policies/handbooks_manuals/aviation'
    main(max_files=5, url=url)  # Limit to 5 files for testing purposes
    