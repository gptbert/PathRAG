# %%
import os
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import siliconflow_complete

WORKING_DIR = ""

api_key = ""
os.environ["SILICONFLOW_API_KEY"] = api_key
base_url = "https://api.siliconflow.cn/v1"
os.environ["SILICONFLOW_BASE_URL"] = base_url


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
