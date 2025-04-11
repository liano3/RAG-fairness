# 🤖RAG-fairness

## 📌Project Overview

Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by retrieving relevant document from external knowledge sources. Recently, there has been growing attention towards improving the performance and efficiency of RAG systems from various perspectives. While these advancements have yielded significant results, the application of RAG in domains with considerable societal implications raises a critical question about fairness: **What impact does the introduction of the RAG paradigm have on the fairness of LLMs?** 

To address this question, we conduct extensive experiments by varying the LLMs, retrievers, and retrieval sources. Our experimental analysis reveals that the scale of the LLMs plays a significant role in influencing fairness outcomes within the RAG framework. For small-scale LLMs (e.g., Llama3.2-1b, Mistral-7b, and Llama3-8b), the integration of retrieval mechanisms often exacerbates unfairness. 


To mitigate the fairness issues introduced by RAG for small-scale LLMs, we propose two approaches, FairFT and FairFilter. Specifically, in FairFT, we align the retriever with the LLM in terms of fairness, enabling it to retrieve documents that facilitate fairer model outputs. In FairFilter, we propose a fairness filtering mechanism to filter out biased content after retrieval. Finally, we validate our proposed approaches on real-world datasets, demonstrating their effectiveness in improving fairness while maintaining performance.

### ⚖️Fairness Evaluation

- **Data Download and Processing:** Provides two types of retrieval libraries, WebPage and Wikipedia, along with the fairness evaluation dataset TrustLLM and utility evaluation dataset CRAG.
- **Retriever:** Includes both sparse retrievers (BM25) and dense retrievers supported by transformers.
- **LLM:** Supports closed-source models like GPT, GLM, Gemini, and other open-source models.
- **Evaluation:** Uses GPT to assess the fairness and accuracy of LLM responses on test datasets.

### 🛠️Fairness Improvement

- **fairFilter:** Adds a filter component between the retriever and LLM to exclude biased documents, reducing the impact of unfair or irrelevant documents on generation quality.
- **fairFT:** Trains retrievers using the ReplugLSR method to align with LLM preferences in terms of fairness, aiming to retrieve documents that make LLM responses fairer.

## ⚙️Installation Steps

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

4. Run scripts:

    - **`config.py`**: This file contains configuration settings used across the project.
    - **`fairFilter.py`**: Generates content, with an optional document filtering mechanism.
    - **`eval.py`**: Performs fairness and accuracy evaluation of the generated content.
    - **`fairFT.py`**: Trains a retriever to improve content fairness.

## ❤️Acknowledgements

We would like to thank the authors of the following works and repositories, which provided valuable resources and inspiration for this project:

- **Fairness Evaluation**
  We refer to the datasets and evaluation code from [TrustLLM](https://github.com/HowieHwong/TrustLLM):
  *Y. Huang et al., "TrustLLM: A Benchmark for Trustworthiness of Large Language Models", arXiv:2401.05561*
  https://arxiv.org/abs/2401.05561
- **Utility Evaluation**
  We use evaluation datasets and code from [CRAG](https://github.com/facebookresearch/CRAG):
  *X. Yang et al., "CRAG: A Benchmark for Evaluating Retrieval-Augmented Generation", arXiv:2406.04744*
  https://arxiv.org/abs/2406.04744
- **Retriever Training**
  Our retriever training implementation is based on [REPLUG](https://github.com/swj0419/REPLUG):
  *W. Shi et al., "REPLUG: Retrieval-Augmented Black-Box Language Models", arXiv:2301.12652*
  https://arxiv.org/abs/2301.12652

We sincerely appreciate the contributions of these researchers and open-source communities.

