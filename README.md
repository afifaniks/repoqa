# RepoQA: Repository-level Question Answering with RAG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](https://hub.docker.com/r/afifaniks/repoqa)
[![Docker Image Size](https://img.shields.io/docker/image-size/afifaniks/repoqa/latest)](https://hub.docker.com/r/afifaniks/repoqa)
[![Docker Pulls](https://img.shields.io/docker/pulls/afifaniks/repoqa)](https://hub.docker.com/r/afifaniks/repoqa)
[![codecov](https://codecov.io/github/afifaniks/repoqa/graph/badge.svg?token=APVRZCE8G9)](https://codecov.io/github/afifaniks/repoqa)

<img src="assets/preview.jpg" width="60%"/>

<small><i>Image generated with ChatGPT</i></small>

RepoQA is a software **repository-level question answering system** powered by **Retrieval-Augmented Generation (RAG)** and a **Large Language Model (LLM)**.  
It allows users to ask natural language questions about a software repository and receive context-aware answers grounded in the repository’s code and documentation.

## Features
- **Repository-level Question Answering**: Ask questions about functions, classes, modules, or overall repository design.
- **Retrieval-Augmented Generation (RAG)**: Combines semantic retrieval with generative reasoning for precise and grounded answers.
- **Code-Aware Retrieval**: Extracts relevant files, functions, and comments from the repository to provide evidence-based responses.
- **LLM Integration**: Uses a large language model to generate human-like, context-rich answers.

## Architecture
1. **Indexing**:  
   - The repository codebase is parsed and embedded using a code-aware embedding model.  
   - Metadata (file paths, function definitions, docstrings) is stored in a vector database.

2. **Query Processing**:  
   - User provides a natural language query.  
   - The system retrieves the most relevant code snippets/files using semantic similarity search.  

3. **Answer Generation**:  
   - The retrieved context is passed into an LLM with the user's question.  
   - The LLM generates a grounded and contextually accurate answer.  

## Quick Start with Docker

The easiest way to get started with RepoQA is using Docker:

### Prerequisites
- Docker or Podman installed
- NVIDIA GPU (optional, but recommended for better performance)
- NVIDIA Container Toolkit (for GPU support)

### Pull and Run

```bash
# Pull the image from Docker Hub
docker pull afifaniks/repoqa:latest

# Run with GPU support (Docker with NVIDIA Container Toolkit)
docker run --gpus all \
  -v $(pwd)/ollama_data:/root/.ollama \
  -p 8000:8000 \
  afifaniks/repoqa:latest

# Run with GPU support (Podman with CDI)
podman run --device nvidia.com/gpu=all \
  -v $(pwd)/ollama_data:/root/.ollama \
  -p 8000:8000 \
  afifaniks/repoqa:latest

# Or run without GPU (Docker)
docker run \
  -v $(pwd)/ollama_data:/root/.ollama \
  -p 8000:8000 \
  afifaniks/repoqa:latest

# Or run without GPU (Podman)
podman run \
  -v $(pwd)/ollama_data:/root/.ollama \
  -p 8000:8000 \
  afifaniks/repoqa:latest
```

**GPU Support Notes:**
- **Docker**: Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and uses `--gpus all` flag
- **Podman**: Uses CDI (Container Device Interface) with `--device nvidia.com/gpu=all` flag
- GPU support significantly improves LLM inference performance

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for the interactive API documentation.

### Using the API

Once the container is running, you can ask questions about repositories:

```bash
# Ask a question about a repository
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "repo": "https://github.com/afifaniks/repoqa.git",
    "question": "How does this application work?",
    "llm_model": "qwen3:1.7b"
  }'
```

## API Documentation

### Endpoints

#### `POST /ask`

Ask a question about a repository.

**Request Body:**

```json
{
  "repo": "string (required) - Repository URL or local path",
  "question": "string (required) - Question about the repository",
  "mode": "string (optional) - 'rag' or 'agent' (default: 'rag')",
  "llm_model": "string (optional) - A valid Ollama LLM model name (default: from config)",
  "force_update": "boolean (optional) - Force re-indexing (default: false)"
}
```

**Response:**

```json
{
  "question": "string - Your question",
  "answer": "string - Generated answer",
  "repo": "string - Repository path/URL"
}
```

**Example Requests:**

```bash
# Basic question with default RAG mode
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "repo": "https://github.com/afifaniks/repoqa.git",
    "question": "What are the permissible licenses in this repository?"
  }'

# Using agent mode for complex questions
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "repo": "https://github.com/afifaniks/repoqa.git",
    "question": "How does the RAG pipeline work and what files implement it?",
    "mode": "agent",
    "llm_model": "qwen3:8b"
  }'

# Force re-indexing with custom model
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "repo": "https://github.com/afifaniks/repoqa.git",
    "question": "Explain the architecture",
    "llm_model": "qwen3:1.7b",
    "force_update": true
  }'
```

#### `GET /`

Health check endpoint.

**Response:**

```json
{
  "status": "OK",
  "service": "RepoQA API",
  "version": "0.1.0"
}
```

#### `GET /health`

Detailed health check endpoint.

**Response:**

```json
{
  "status": "healthy"
}
```

### Request Modes

RepoQA supports two modes of operation, each optimized for different use cases:

#### **RAG Mode** (Default)

**When to use:**
- **When you don't have access to a suitable GPU**
- Simple, direct questions about repo/code functionality
- Questions that can be answered with semantic search
- Fast responses needed
- Lower computational requirements

**How it works:**
- Uses pure Retrieval-Augmented Generation (RAG)
- Performs semantic similarity search on indexed code
- Retrieves top-k relevant documents
- Generates answer based on retrieved context
- Single-pass processing

---

#### **Agent Mode (Experimental)**

**When to use:**
- **If you have access to GPU**
- Complex questions requiring multi-step reasoning
- Questions about file structure and organization
- Questions requiring directory navigation
- Debugging and tracing code flow across files

**How it works:**
- Uses ReAct (Reasoning + Acting) agent framework
- Has access to multiple tools:
  - `semantic_search`: Find relevant code via similarity
  - `similarity_search_with_score`: Search with relevance scores
  - `list_directory`: Explore repository structure
  - `read_file`: Read complete file contents
- Agent iteratively uses tools to gather information
- Reasons about what information is needed
- Can explore repository structure dynamically

_[This mode is still under development]_

---


### Interactive API Documentation

Visit `http://localhost:8000/docs` for the interactive Swagger UI documentation where you can:
- Try out API endpoints directly from your browser
- See detailed request/response schemas
- View example requests and responses
- Test different modes and parameters


### Building from Source

If you want to build the Docker image yourself:

```bash
# Clone the repository
git clone https://github.com/afifaniks/repoqa.git
cd repoqa

# Build with Docker
docker build -t repoqa .

# Build with Podman
podman build -t repoqa .
```

**Running the Built Image:**

```bash
# Docker with GPU support (requires NVIDIA Container Toolkit)
docker run --gpus all \
  -v $(pwd)/ollama_data:/root/.ollama \
  -p 8000:8000 \
  repoqa

# Docker without GPU
docker run \
  -v $(pwd)/ollama_data:/root/.ollama \
  -p 8000:8000 \
  repoqa

# Podman with GPU support (uses CDI)
podman run --device nvidia.com/gpu=all \
  -v $(pwd)/ollama_data:/root/.ollama \
  -p 8000:8000 \
  repoqa

# Podman without GPU
podman run \
  -v $(pwd)/ollama_data:/root/.ollama \
  -p 8000:8000 \
  repoqa
```

### Docker Compose

For easier management, use docker-compose:

```bash
# Start the service
docker-compose up

# With custom settings
REPO_PATH="https://github.com/your-repo.git" \
QUESTION="Your question here" \
docker-compose up
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- How to set up the development environment
- Our coding standards and style guide
- How to submit changes and report bugs
- Where to ask for help

To see the current status of the project, please have a look at [RepoQA Kanban Board](https://github.com/users/afifaniks/projects/9/).

### Quick Start for Contributors

```bash
# Clone your fork
git clone https://github.com/<your-username>/repoqa.git
cd repoqa

# Set up development environment
make setup

# Run tests to verify everything works
make test

# See all available commands
make help
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### OSS Component Usage Policy

All dependencies are checked for license compatibility. See our:
- [NOTICE](NOTICE) file for third-party attributions  
- [OSS Component Usage Policy](CONTRIBUTING.md#oss-component-usage-policy) for contribution guidelines
- [License Policy Quick Reference](docs/license-policy.md) for allowed/prohibited licenses

## Generative AI Usage
GitHub Copilot with Claude Sonnet 4 was used to generate boilerplates, documentation, test cases for this project. However, all the generated contents were manually validated.
