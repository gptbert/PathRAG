# -*- coding: utf-8 -*-
# %%
import os
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import siliconflow_complete

WORKING_DIR = "kg"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = PathRAG(
    working_dir=WORKING_DIR,
    llm_model_func=siliconflow_complete,
)

# %%
data_file = "data/Anthropic.md"
with open(data_file) as f:
    rag.insert(f.read())

# %%
question = "用简体中文回答，Agent如何优化响应以提高 token 效率？"
print(rag.query(question, param=QueryParam(mode="hybrid")))
