import os
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import siliconflow_complete

WORKING_DIR = ""

api_key = ""
os.environ["OPENAI_API_KEY"] = api_key
base_url = "https://api.siliconflow.cn/v1"
os.environ["OPENAI_BASE_URL"] = base_url


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = PathRAG(
    working_dir=WORKING_DIR,
    llm_model_func=siliconflow_complete,
)

data_file = ""
question = ""
with open(data_file) as f:
    rag.insert(f.read())

print(rag.query(question, param=QueryParam(mode="hybrid")))
