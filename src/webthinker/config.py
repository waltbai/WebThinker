"""Config."""

# Paths
NLTK_DATA_PATH = "./thirdparty/nltk_data"

# Model
BASEURL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
PLANNER_MODEL = "qwen3-32b"
SUPERVISOR_MODEL = "qwen3-32b"
WRITER_MODEL = "qwen3-32b"
RESEARCHER_MODEL = "qwen3-32b"
SEED = 42

# Supervisor
MAX_INTERACTIONS = 20
MAX_OUTPUT_RETRY = 3

# Search Query Tool
SEARCH_TOP_K = 10
