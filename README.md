## KGGen Examples

This codebase has been modified from Mo, Belinda, et al. "Kggen: Extracting knowledge graphs from plain text with language models." arXiv preprint arXiv:2502.09956 (2025).

This project generates knowledge graphs from text using LLMs and deduplication tools, with HTML visualization.

### Prerequisites
- Python 3.10+
- Access to an LLM provider supported by `litellm` (e.g., OpenAI, Azure, etc.)

### Installation
1. Create a virtualenv (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

### `.env` Configuration
Create a `.env` file in the project root with these variables:

```dotenv
LLM_MODEL="openai/gpt-4o"
LLM_API_KEY="YOUR_API_KEY"
LLM_TEMPERATURE="0.0"
API_BASE=""
RETRIEVAL_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```

To use a local Ollama model:
```dotenv
LLM_MODEL="ollama_chat/qwen3:14b"
LLM_API_KEY=""
API_BASE="http://localhost:11434"
```

Optional variables to silence Hugging Face logs:
```dotenv
TRANSFORMERS_VERBOSITY="error"
HF_HUB_DISABLE_PROGRESS_BARS="1"
HF_HUB_DISABLE_TELEMETRY="1"
TOKENIZERS_PARALLELISM="true"
TQDM_DISABLE="1"
```

Notes:
- `LLM_MODEL` is the model identifier for `litellm`.
- `LLM_API_KEY` is your provider API key.
- `LLM_TEMPERATURE` must be a float.
- `API_BASE` is optional (useful for Azure/OpenAI-compatible endpoints).
- `RETRIEVAL_MODEL` is the SentenceTransformer model used for retrieval.

### Running
Run `run.py` from the project root:

```bash
python run.py
```

Output:
- Timings and graphs are printed to the console.
- HTML files are generated and opened automatically in your browser.
