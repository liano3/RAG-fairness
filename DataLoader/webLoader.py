from bs4 import BeautifulSoup
import bz2
import json
import os
from tqdm import tqdm
import requests
import time
from concurrent.futures import ThreadPoolExecutor

import config

def extract_dataset(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Read the CRAG dataset
    with bz2.open("corpus/web/crag_task_1_dev_v4_release.jsonl.bz2", "rt", encoding='utf-8') as f:
        for line in tqdm(f):
            data = json.loads(line)
            # Save query, answer, alt_ans
            data1 = {
                "query_time": data["query_time"],
                "query": data["query"],
                "domain": data["domain"],
                "question_type": data["question_type"],
                "static_or_dynamic": data["static_or_dynamic"],
                "answer": data["answer"],
                "alt_ans": data["alt_ans"]
            }
            with open(os.path.join(save_path, 'query.jsonl'), "a", encoding='utf-8') as f1:
                f1.write(json.dumps(data1) + "\n")

            # Uncomment the following lines if you need to extract HTML
            # data2 = data["search_results"]
            # for html in data2:
            #     extract_html(html["page_result"])

def extract_html(html_source, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    soup = BeautifulSoup(html_source, 'lxml')
    # Extract text
    text = soup.get_text()
    text = ' '.join(text.split())
    # Save text
    with open(os.path.join(save_path, 'crag.txt'), "a", encoding='utf-8') as f:
        f.write(text + "\n")

# Split the dataset into 10 parts to make it easier to process
def split_dataset():
    with open("corpus/web/crag.txt", "r", encoding='utf-8') as f:
        lines = f.readlines()
    # Remove duplicate lines
    lines = list(set(lines))
    for i in range(10):
        with open(f"corpus/web/txt/crag_{i}.txt", "w", encoding='utf-8') as f:
            f.writelines(lines[i * len(lines) // 10: (i + 1) * len(lines) // 10])
    # Remaining part
    with open(f"corpus/web/txt/crag_10.txt", "w", encoding='utf-8') as f:
        f.writelines(lines[10 * len(lines) // 10:])

def resume_query():
    if not os.path.exists("api_res.jsonl"):
        print("Start from scratch.")
        return set()
    visited_query = set()
    with open("api_res.jsonl", "r", encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            visited_query.add(data["query"]["original"])
    print(f"{len(visited_query)} queries have been visited. Resume from here.")
    return visited_query

# Use Brave Web Search API to get web pages related to the query
def get_web_page(query):
    base_url = f"https://api.search.brave.com/res/v1/web/search"
    headers = {
        "X-Subscription-Token": config.brave_api_key
    }
    response = requests.get(base_url, headers=headers, params={"q": query, "count": 20, "result_filter": "web"})
    data = response.json()
    if data["type"] == "ErrorResponse":
        time.sleep(1)
        get_web_page(query)
        return
    with open("api_res.jsonl", "a", encoding='utf-8') as f:
        f.write(json.dumps(data) + "\n")

def resume_url():
    if not os.path.exists("web_page.jsonl"):
        return set()
    visited_url = set()
    with open("web_page.jsonl", "r", encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
            except:
                print(line[0:100])
                continue
            visited_url.add(data["url"])
    print(f"{len(visited_url)} URLs have been visited. Resume from here.")
    return visited_url

def crawl(url):
    headers = {
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0",
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return url, False
        with open("web_page.jsonl", "a", encoding='utf-8') as f:
            f.write(json.dumps({"url": url, "html": response.text}) + "\n")
        return url, True
    except:
        return url, False

def get_all_urls(file):
    with open(file, "r", encoding='utf-8') as f:
        lines = f.readlines()
    urls = set()
    for line in tqdm(lines):
        data = json.loads(line)
        try:
            data = data["web"]["results"][:5]
        except:
            continue
        for item in data:
            urls.add(item["url"])
    return urls

def crawl_all(all_urls):
    visited_url = resume_url()
    urls = all_urls - visited_url
    with ThreadPoolExecutor(max_workers=10) as executor:
        for _, _ in tqdm(executor.map(crawl, urls), total=len(urls)):
            pass

def extract_web_page():
    with open("web_page.jsonl", "r", encoding='utf-8') as f:
        lines = f.readlines()
    for line in tqdm(lines):
        try:
            data = json.loads(line)
        except:
            continue
        extract_html(data["html"])

def get_fairness_web_page(path):
    visited_query = resume_query()
    # Read the fairness dataset and use sentences as queries to get web pages
    with open(os.path.join(path, "stereotype_recognition.json"), "r", encoding='utf-8') as f:
        data = json.load(f)
    with open(os.path.join(path, "stereotype_agreement.json"), "r", encoding='utf-8') as f:
        data += json.load(f)
    data = [item["refined_query"] for item in data]
    for query in tqdm(data):
        if query in visited_query:
            continue
        get_web_page(query)

# Split file into n parts
def split_file(file, n):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        num_lines = len(lines)
    print(f"Total lines: {num_lines}")
    part_size = num_lines // n
    for i in range(n):
        new_file = file.split('.')[0] + f"_{i}" + '.' + file.split('.')[1]
        with open(new_file, 'w', encoding='utf-8') as part:
            part.writelines(lines[i * part_size: (i + 1) * part_size])
    new_file = file.split('.')[0] + f"_{n}" + '.' + file.split('.')[1]
    with open(new_file, 'w', encoding='utf-8') as part:
        part.writelines(lines[n * part_size:])
    print(f"Split into {n} parts")

if __name__ == "__main__":
    # extract_dataset()
    # split_dataset()
    get_fairness_web_page()
    all_urls = set()
    for i in range(4):
        all_urls |= get_all_urls(f"api_res.jsonl")
    print(f"Total URLs: {len(all_urls)}")
    crawl_all(all_urls)
    extract_web_page()
