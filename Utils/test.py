import os
import json
import re
import matplotlib.pyplot as plt
from collections import Counter
import random
import pickle

def get_doc(name):
    with open('results/generations/fairness/stereotype_agreement_' + name + '.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    data = [{'prompt': d['prompt'], 'docs': d['docs']} for d in data]
    with open('results/fairness_' + name + '.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    with open('results/generations/crag/crag_' + name + '.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    data = [{'query_time': d['query_time'], 'query': d['query'], 'docs': d['docs'], 'answer': d['answer'], 'alt_ans': d['alt_ans']} for d in data]
    with open('results/crag_' + name + '.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def remove_doc():
    for root, dirs, files in os.walk('results/generations/fairness'):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                data = [{'prompt': d['prompt'], 'res': d['res']} for d in data]
                with open(os.path.join('results/generations/fairness', file), 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)

    for root, dirs, files in os.walk('results/generations/crag'):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                data = [{'query': d['query'], 'answer': d['answer'], 'alt_ans': d['alt_ans'], 'res': d['res']} for d in data]
                with open(os.path.join('results/generations/crag', file), 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)

def find_case(name1, name2):
    with open('stereotype_agreement_' + name1 + '.json', 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    with open('stereotype_agreement_' + name2 + '.json', 'r', encoding='utf-8') as f:
        data2 = json.load(f)

    data1 = [d for d in data1 if d['eval_res'] == 'yes']
    data2 = [d for d in data2 if d['eval_res'] == 'no']

    data = []
    for d1 in data1:
        for d2 in data2:
            if d1['prompt'] == d2['prompt']:
                sentence = re.findall(r'\"(.*?)\"', d1['prompt'])[0]
                data.append(sentence)

    with open('case.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def clean_output(name):
    path = 'results/generations/fairness/stereotype_agreement_' + name + '.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for d in data:
            d['res'] = d['res'].split('assistant')[0]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def plot_bar(label, value, title):
    fig, ax = plt.subplots(figsize=(3, 2), constrained_layout=True)
    ax.bar(label, value, color='#4c72b0')
    ax.set_title(title, fontsize=15, loc='center', pad=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', left=False, labelleft=False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.show()

def count_type(name):
    with open('datasets/fairness/stereotype_agreement.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    counter = Counter([d['bias_type'] for d in data])
    print(counter)
    prompt_bias = {}
    for d in data:
        prompt_bias[d['prompt']] = d['bias_type']
    print("=====================================")

    with open('datasets/crag/query-plus.jsonl', 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            data.append(json.loads(line))

    counter = Counter([d['domain'] for d in data])
    print(counter)
    query_domain = {}
    for d in data:
        query_domain[d['query']] = d['domain']
    print("=====================================")

    path0 = "results/generations/fairness/stereotype_agreement_" + name + ".json"
    with open(path0, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for d in data:
            d['bias_type'] = prompt_bias[d['prompt']]
    with open(path0, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    path1 = "results/generations/crag/crag_" + name + ".json"
    with open(path1, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for d in data:
            d['domain'] = query_domain[d['query']]
    with open(path1, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def get_train_data():
    with open('wiki_doc.pkl', 'rb') as f:
        wiki_doc = pickle.load(f)

    with open('doc.pkl', 'rb') as f:
        doc = pickle.load(f)

    print(len(wiki_doc))
    random.seed(42)
    sampled_wiki_doc = random.sample(wiki_doc, len(wiki_doc) // 20)
    train_doc = sampled_wiki_doc + doc
    print(train_doc[0:5])
    with open('train_doc.pkl', 'wb') as f:
        pickle.dump(train_doc, f)

def parse_res(raw):
    resp = re.search(r"\{.*?\}", raw, re.DOTALL).group(0)
    resp = resp.lower()
    model_resp = json.loads(resp)
    return model_resp

def merge_log(name, name1, name2):
    data = []
    with open(name1, "r", encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    with open(name2, "r", encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    with open(name, "w", encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
