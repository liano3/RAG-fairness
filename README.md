# RAG-fairness

## Project Overview

RAG-fairness aims to evaluate and enhance the fairness of Retrieval-Augmented Generation (RAG) for Large Language Models (LLMs). We are dedicated to improving fairness in the RAG paradigm while minimizing any impact on performance. The project mainly involves the following aspects:

### Fairness Evaluation

- **Data Download and Processing:** Provides two types of corpora, WebPage and Wikipedia, along with the fairness evaluation dataset TrustLLM and utility evaluation dataset CRAG.
- **Retriever Construction and Usage:** Includes both sparse retrievers (BM25) and dense retrievers supported by transformers.
- **LLM Construction and Usage:** Supports closed-source models like GPT, GLM, Gemini, and other open-source models.
- **RAG Construction and Usage:** Integrates retrievers and LLMs into a RAG system to enhance generation using retrieved documents.
- **Evaluation:** Uses GPT to assess the fairness and accuracy of LLM responses on test datasets.

### Fairness Improvement

- **Filter Implementation:** Adds a filter component between the retriever and LLM to exclude biased documents, reducing the impact of unfair or irrelevant documents on generation quality.
- **Retriever Training:** Trains retrievers using the ReplugLSR method to align with LLM preferences, aiming to retrieve documents that make LLM responses fairer.

## Installation Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/liano3/RAG-fairness.git
    ```

2. Enter the project directory:

    ```bash
    cd RAG-fairness
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Project Directory Structure

```
RAG-fairness/ 
├── Component/ 
│ ├── bm25Retriever.py 
│ ├── gptLLM.py 
│ ├── hfLLM.py 
│ ├── hfRetriever.py 
├── DataLoader/ 
│ ├── webLoader.py 
│ ├── wikiLoader.py 
├── Main/ 
│ ├── eval.py 
│ ├── gen.py 
│ ├── replug_lsr.py 
├── Utils/ 
│ ├── bar.py 
│ ├── test.py 
│ ├── trade-off.py 
├── README.md 
├── config.py 
├── requirements.txt
```

## Usage Instructions

1. **Configure the Project:**
   - Edit the `config.py` file to set the required parameters and configurations.
2. **Data Loading:**
   - Use the scripts in the `DataLoader` folder to load and process the necessary data.
3. **Run Main Functions:**
   - Use the scripts in the `Main` folder to execute the main functions.
4. **Other Tools:**
   - Use the tools in the `Utils` folder to organize results and visualize through plotting.