# GWI – Jedi Team – AI Engineering Challenge

Welcome to the **AI engineering** challenge for the Jedi Team at GWI!

This task is designed to help us understand how you approach applied‐AI problems and build intelligent, production‑ready systems. The Jedi team owns and evolves the company’s AI infrastructure, so the exercise focuses on agentic patterns, tool orchestration, and reasoning quality. Creativity, thoughtful design, and clean code are all appreciated.

---

##    How to Run

<pre> ```bash # Create virtual environment python -m venv venv # Activate it (Linux/macOS) source venv/bin/activate # Install dependencies pip install -r requirements.txt ``` </pre>
---

## 🧪 Core Scenario

GWI’s market‑research data has been converted into natural‑language statements and stored in a relational database. A sample export lives in `data.md`.

Your mission is to implement an **agentic chatbot** that helps clients answer questions using that data—and, when needed, additional tools.

### The agent should be able to

1. **Retrieve answers from internal data**  
   * Exact match or similarity search is acceptable.  
   * Cite the data fragment you used.

2. **Plan a course of action** when the answer is not immediately available.  
   * E.g. “I will check the database; if that fails, I’ll run a web search.”

3. **Call at least two external tools** (examples, choose any two or invent your own):  
   * Web‑search API.  
   * Vector‑database or RAG lookup.  
   * Classifier to ensure quality of the answer.

4. **Evaluate and Score Agent Responses**  
   * Implement an evaluation process to assess the quality and relevance of the agent's answers.  
   * Use the evaluation results to iteratively improve the agent's reasoning and tool selection.

5. **Finetune the Agent Based on Feedback**  
   * Incorporate user feedback and evaluation outcomes to continuously finetune the agent's models or decision logic.  
   * This may involve retraining components, updating prompts, or refining tool selection strategies to enhance performance over time.

6. **Persist conversations** so a user can resume from where they left off.  
   * Support multiple concurrent chats per user.

7. **Expose an HTTP interface** (REST, SSE, or WebSocket) that allows a UI to stream the agent’s intermediate reasoning (“thoughts”) and final answers.

You may implement the service in **Go 1.22+ or Python 3.11+**. Mixing in other languages or frameworks for tooling is fine if it helps.

---

## 🌟 Nice‑to‑have Enhancements

* Auto‑generate a concise, human‑readable title for every new chat.  
* Allow users to give thumbs‑up / thumbs‑down feedback on any agent message.  
* Log tool invocations and surface simple analytics (e.g., % of questions resolved without web search).  
* Containerisation (Docker) or a dev script (`make`, `task`, `invoke`, etc.) for one‑command setup.  
* Deployment notes for a cloud environment of your choice.  
* AI‑generated simple UI (100 % optional).

---

## 🚀 Getting Your Solution Running

Provide a short guide (README section or shell script) that covers:

1. **Startup** – how to launch the service (and any supporting services such as a vector DB).  
2. **Configuration** – environment variables, API keys, or model endpoints needed.  
3. **Testing** – how to run unit and integration tests.
4. **Evaluation** – how to evaluate the performance of the agent.
5. **Assumptions** – anything non‑obvious you decided (e.g., which LLM provider you picked, why a specific vector DB).

---

## 🧩 Submission

Fork this repository, commit your solution to your fork, and send it back to us.

Good luck, potential colleague—may the (reinforcement) learning be with you!
