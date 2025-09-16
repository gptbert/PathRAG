# PathRAG

Code accompanying the paper "PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths".

## Overview
PathRAG builds a relational graph over your knowledge base so that a language model can answer questions by walking the most relevant paths. The library focuses on:
- pruning noisy connections while retaining semantically meaningful paths,
- combining graph traversal with hybrid semantic-keyword retrieval,
- offering a lightweight interface for rapid experimentation with new LLMs or corpora.

## Installation
```bash
cd PathRAG
pip install -e .
```

## Prepare Your Environment
1. Create or choose a working directory where PathRAG can store intermediate assets.
2. Provide a SiliconFlow (OpenAI-compatible) API endpoint and key if you plan to use the default LLM helper.

You can set the required environment variables directly in Python before instantiating `PathRAG`:
```python
import os

os.environ["OPENAI_API_KEY"] = "your_api_key"
os.environ["OPENAI_BASE_URL"] = "https://api.siliconflow.cn/v1"
```

## Quick Start
The `v1_test.py` script demonstrates the main workflow. The essential steps are distilled below:
```python
import os
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import siliconflow_complete

WORKING_DIR = "./your_working_dir"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = PathRAG(
    working_dir=WORKING_DIR,
    llm_model_func=siliconflow_complete,
)

data_file = "./text.txt"
question = "your_question"

with open(data_file, encoding="utf-8") as f:
    rag.insert(f.read())

answer = rag.query(question, param=QueryParam(mode="hybrid"))
print(answer)
```
### Notes
- Replace `data_file` with the document you want to index prior to querying.
- `QueryParam(mode="hybrid")` combines semantic and graph retrieval; explore other modes in `operate.py`.
- Swap `siliconflow_complete` for any callable that matches the expected LLM interface.

## Configuration
Most tuning knobs live in `base.py` and `operate.py`. Key areas include:
- graph construction thresholds,
- retrieval depth and breadth limits,
- scoring and reranking strategies.

Adjust these values to balance recall, precision, and runtime for your dataset.

## Batch Ingestion
For large corpora, iterate over a folder of text files:
```python
import os

folder_path = "your_folder_path"
rag = ...  # instantiate PathRAG as above

txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
for file_name in txt_files:
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, encoding="utf-8") as file:
        rag.insert(file.read())
```

## Citing PathRAG
Please cite the paper if you build on this project:
```bibtex
@article{chen2025pathrag,
  title={PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths},
  author={Chen, Boyu and Guo, Zirui and Yang, Zidan and Chen, Yuluo and Chen, Junze and Liu, Zhenghao and Shi, Chuan and Yang, Cheng},
  journal={arXiv preprint arXiv:2502.14902},
  year={2025}
}
```
