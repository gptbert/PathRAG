# -*- coding: utf-8 -*-
# %%
import os
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import siliconflow_complete

WORKING_DIR = "kg"

os.environ["OPENAI_API_KEY"] = os.getenv("SILICONFLOW_API_KEY", "")
os.environ["OPENAI_BASE_URL"] = os.getenv("SILICONFLOW_BASE_URL", "")


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
question = "Anthropic为Agent构建工具的工程流程是？"
print(rag.query(question, param=QueryParam(mode="hybrid")))

# %%
