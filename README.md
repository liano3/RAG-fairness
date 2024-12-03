# RAG-fairness

## Project Overview

RAG-fairness is a project aimed at evaluating and improving the fairness of Retrieval-Augmented Generation (RAG). It mainly implements the following aspects:

- Fairness Evaluation

  - Data Download and Processing: Provides two types of corpora, WebPage and Wikipedia, as well as fairness evaluation dataset TrustLLM and performance evaluation dataset CRAG.

  - Retriever Construction and Usage: Includes sparse retrievers (BM25) and other dense retrievers supported by transformers.

  - LLM Construction and Usage: Includes closed-source models like GPT, GLM, Gemini, and other open-source models supported by transformers.

  - RAG Construction and Usage: Simply assembles retrievers and LLMs into a RAG system to enhance generation with retrieved documents.
  - Evaluation: Uses GPT to evaluate the fairness and accuracy of LLM responses on test datasets.

- Fairness Improvement
  - Implementation of Filter: Adds a Filter component between the retriever and LLM to filter out biased documents, reducing the impact of unfair or irrelevant documents on generation quality.
  - Retriever Training: Trains retrievers using the ReplugLSR method to align with LLM preferences, aiming to retrieve documents that make LLM responses fairer.

## Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/liano3/RAG-fairness.git

1. Enter the project directory:

   ```bash
   cd RAG-fairness
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Project Directory

```
RAG-fairness/
├── Component/
│   ├── bm25Retriever.py
│   ├── gptLLM.py
│   ├── hfLLM.py
│   ├── hfRetriever.py
├── DataLoader/
│   ├── webLoader.py
│   ├── wikiLoader.py
├── Main/
│   ├── eval.py
│   ├── gen.py
│   ├── replug_lsr.py
├── Utils/
│   ├── bar.py
│   ├── test.py
│   ├── trade-off.py
├── README.md
├── config.py
├── requirements.txt
```

## Usage Instructions

1. Configure the project:
   - Edit the `config.py` file to set the required parameters and configurations.
2. Data Loading:
   - Use the code in the `DataLoader` folder to load and process needed data.
3. Run Main Functions:
   - Use the scripts in the `Main` folder to execute main functions.
4. Other Tools:
   - Use the tools in the `Utils` folder to organize results and visualize through plotting.