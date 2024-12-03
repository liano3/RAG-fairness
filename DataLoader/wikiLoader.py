import os
from sentence_transformers import SentenceTransformer
import zipfile
import shutil
import requests
from tqdm import tqdm
import re

# Extract texts from Wikipedia content
def extract_text(content):
    docs = re.findall(r'<doc[^>]*>(.*?)</doc>', content, re.DOTALL)
    result = []
    for doc in docs:
        doc = doc.strip()
        lines = doc.splitlines()
        lines = [line.strip() for line in lines if line.strip()]  # Skip empty lines
        
        if len(lines) > 1:
            body = '\n'.join(lines[1:])  # Skip the first line (title), keep the rest (body)
            body = body.replace('\n', '  ')
            result.append(body)
    
    return result

# Download Sentence-Transformers model
def download_retrieve(model_name, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_path = os.path.join(save_path, model_name.replace("/", "-"))
    print(f"Downloading {model_name}...")
    model = SentenceTransformer(model_name)
    model.save(model_path)

# Download TrustLLM dataset
def download_dataset(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("Downloading datasets...")
    os.system(f"wget https://github.com/HowieHwong/TrustLLM/raw/main/dataset/dataset.zip -P {save_path}")

    # Unzip dataset
    with zipfile.ZipFile(os.path.join(save_path, "dataset.zip"), "r") as z:
        z.extractall(save_path)
    os.remove(os.path.join(save_path, "dataset.zip"))
    # Keep only the fairness folder and delete the others
    for f in os.listdir(save_path):
        if f != "fairness":
            shutil.rmtree(os.path.join(save_path, f))

# Download file with progress bar and support for resuming downloads
def download_file(url, local_filename):
    # Check if the file already exists
    if os.path.exists(local_filename):
        downloaded_size = os.path.getsize(local_filename)
    else:
        downloaded_size = 0
    
    # Send a HEAD request to get the file size
    response = requests.head(url)
    file_size = int(response.headers.get('content-length', 0))
    if downloaded_size < file_size:
        # Set the headers for resuming the download
        headers = {"Range": f"bytes={downloaded_size}-"}
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        # Download the file with a progress bar
        progress = tqdm(total=file_size, initial=downloaded_size, unit='B', unit_scale=True, desc=local_filename)
        with open(local_filename, 'ab') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive new chunks
                    file.write(chunk)
                    progress.update(len(chunk))
        progress.close()
    else:
        print(f"{local_filename} already exists")

# Download Wikipedia dump
def download_wikipedia(save_path):
    save_path = os.path.join(save_path, "wikipedia")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("Downloading Wikipedia...")
    url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
    local_filename = os.path.join(save_path, "enwiki-latest-pages-articles.xml.bz2")
    download_file(url, local_filename)

# Extract Wikipedia text
def extract_wikipedia(file_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    progress = tqdm(total=len(os.listdir(file_path)), desc="Processing Files")
    for idx, filename in enumerate(os.listdir(file_path)):
        with open(os.path.join(save_path, filename + ".txt"), "w", encoding='utf-8') as f_out:
            with open(os.path.join(file_path, filename), "r", encoding='utf-8') as f_in:
                text = f_in.read()
                docs = extract_text(text)
                for doc in docs:
                    # Skip short docs
                    if len(doc) < 1000:
                        continue
                    f_out.write(doc + "\n")
                progress.update(1)
    progress.close()

if __name__ == "__main__":
    # download_retrieve()
    # download_dataset()
    # download_wikipedia()
    # from wikiextractor import WikiExtractor
    # python3 WikiExtractor.py -b 256M -o ~/.../extracted ~/.../.xml.bz2
    extract_wikipedia(file_path="corpus/wiki/AA", save_path="corpus/wiki/txt")
