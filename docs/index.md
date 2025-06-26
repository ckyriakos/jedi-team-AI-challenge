# 🤖 Local LLM Agent with FastAPI + LangChain + Ollama

This repo showcases a lightweight, privacy-focused AI agent framework powered by **FastAPI**, **LangChain**, and **Ollama**, running locally with `llama3.1:8b`. It uses guardrails, embeddings, a local vector store (FAISS), and streaming via SSE to simulate "reasoning" without relying on external APIs.

> ✅ **Goal**: Showcase the abilities of small LLMs on modest hardware, while keeping the system scalable, extensible, cost-effective, and private-by-default.

---

### 🧠 Why LangChain?

Used the **LangChain** library due to its rich ecosystem and integrations with tools like:

* **LangGraph** for orchestrating agent flows
* **LangSmith** for observability and evaluation

These tools support a complete **AIOps lifecycle**, from prototyping to scaling in production.

---

## 🔧 How to Install & Run

There are two ways to get started — either using Docker (recommended) or manually with Python.

### ✅ Option 1: Run with Docker (Recommended)

This spins up both the FastAPI app and the `llama3.1:8b` model (via Ollama) in a shared Docker network.

```bash
git clone https://github.com/yourname/yourrepo.git
cd yourrepo
docker-compose up --build
```

* Access the API: [http://localhost:8000/docs](http://localhost:8000/docs)
* Ollama runs at: `http://localhost:11434`

---

### 🐍 Option 2: Run Locally with Python

Use this if you want full control or prefer not to use Docker.

1. **Install dependencies**

```bash
ollama run llama3:8b  # Pulls and warms up the model
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. **Run the FastAPI server**

```bash
uvicorn app.main:app --reload
```

---

## 📦 Project Structure

```bash
.
├── app/
│   ├── main.py              # FastAPI entrypoint
│   ├── prompts/             # Prompt templates and chains
│   ├── utils/               # Guardrails, filters, feedback
│   ├── vectorstore/         # FAISS + embedding setup
│   └── tests/               # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🧠 Features Overview

Using LangChain’s wrappers for HuggingFace and Ollama:

1. **LLM**: Installed and configured `ollama` locally. Selected `llama3.1:8b` for its performance-to-token ratio.
2. **Embeddings**: Used **Nomic Embed** via HuggingFace with remote inference for speed.
3. **Reasoning**: Since `llama3.1:8b` doesn’t expose step-by-step reasoning like GPT or Claude, simulated thought processes using structured prompts + if/else logic.
4. **Guardrails**: Skips unproductive requests (e.g., greetings) with predefined responses.
5. **Vector DB**: Used **FAISS** for in-memory semantic and metadata filtering. It’s lightweight, fast, and ideal for local setups.
6. **API Server**: FastAPI streams thoughts and answers using **Server-Sent Events (SSE)**.
7. **Evaluation**: Logged performance metrics with optional LangChain evaluation hooks.
8. **Feedback Loop**: Basic thumbs-up/down system to log and improve responses.
9. **UX Prompts**: Prompts are designed to guide and engage the user proactively.
10. **Unit Tests**: Tested key utilities like filters and response validators.
11. **UI Notes**: Initially used GPT-generated UI, realized it’s faster to learn a frontend framework! 😂 For production, tools like **Chainlit** or **OpenWebUI** are better choices.

---

## 🐳 Docker Overview

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: "3.9"

services:
  fastapi-app:
    build: .
    ports:
      - "8000:8000"
    networks:
      - ollama-network
    depends_on:
      - ollama

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    networks:
      - ollama-network
    volumes:
      - ollama-data:/root/.ollama
    restart: always

networks:
  ollama-network:

volumes:
  ollama-data:
```

---

## 📚 Documentation

GitHub Pages version available here:
👉 [**View Online Docs**](https://<your-username>.github.io/<your-repo-name>/)

---

## 🔍 Possible Improvements

* [ ] Frontend using React, Svelte, or Chainlit
* [ ] Logging with LangSmith or OpenTelemetry
* [ ] Use LangGraph for reasoning trace visualization
* [ ] Add streaming feedback metrics dashboard
* [ ] Replace FAISS with Chroma or Weaviate for more scale

---

## 🙌 Acknowledgements

* LangChain team for tooling and guidance
* Ollama for enabling local LLMs with a one-liner
* HuggingFace for open embeddings and models

---

## 📄 License

MIT – see [`LICENSE`](LICENSE)

