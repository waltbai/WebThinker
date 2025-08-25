"""Config."""

from typing import Literal

# Paths
NLTK_DATA_PATH = "./thirdparty/nltk_data"

# Model
BASEURL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
PLANNER_MODEL = "qwen2.5-32b-instruct"
SUPERVISOR_MODEL = "qwq-32b"
WRITER_MODEL = "qwen2.5-32b-instruct"
EVALUATION_MODEL = "qwen2.5-72b-instruct"
SEED = 42
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
REPETITION_PENALTY = 1.05

# Supervisor
MAX_INTERACTIONS = 20
MAX_OUTPUT_RETRY = 3

# Search Query Tool
SEARCH_TOOL: Literal["tavily", "google"] = "google"
SEARCH_TOP_K = 10
MAX_SEARCH_LIMIT = 20

# Evaluation
GROUP_KEYS = [
    "level",    # GAIA
    "high-level domain",    # GPQA
    "category", # HLE
    "domain",   # WebWalkerQA
]
