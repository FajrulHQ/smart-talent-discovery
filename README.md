# Smart Talent Discovery

Smart Talent Discovery is an AI-based platform designed for efficient resume and portfolio retrieval. The system leverages image embeddings for document retrieval using Colpali and Gemma3, allowing for fast and accurate candidate searches.

### Features

- AI-powered Resume Retrieval: Uses advanced embedding techniques to process and index resumes.
- Portfolio Analysis: Extracts insights from portfolios based on image embeddings.
- Colpali Integration: Utilizes Colpali for visual document retrieval.
- Gemma3 for QA: Implements Gemma3 for multimodal question-answering (qa).
- Fast Search Capabilities: Enables quick and precise resume or portfolio for talent discovery.

### Installation

**Prerequisites**

Ensure you have the following installed:
- Python 3.8+
- Docker

### Setup

**Clone the repository:**

```zsh
git clone https://github.com/FajrulHQ/smart-talent-discovery.git
cd smart-talent-discovery
```

**Create and activate a virtual environment:**
```zsh
python -m venv .venv
source .venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

**Install dependencies:**
```zsh
pip install -r requirements.txt
```

**Install Ollama**

- download in official website
https://ollama.com/download/

- or using homebrew
```zsh
brew install --cask ollama
```

**Download Colpali models from hugingface**
```zsh
pip install -U "huggingface_hub[cli]"

# hf authentication
huggingface-cli login

# download colpali model to local
huggingface-cli download vidore/colpali-v1.3-hf --local-dir ./models/vidore/colpali-v1.3-hf 
```

**Run Milvus in local**
```zsh
docker compose up
```

### Usage

-- on development..
