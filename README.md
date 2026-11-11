# LLM Agents

## What's in this project

1. [`main.py`](./main.py) demonstrates calling the OpenAI API to classify a document and return structured output.
2. [`transformers/`](./transformers/) contains experiments for learning transformer basics, including tokenization, text generation, and model evaluation.
3. [`rag_demo/`](./rag_demo/) demonstrates retrieval-augmented generation (RAG): embedding documents, retrieving relevant passages, and using them to answer a question.
4. [`agent_demo/`](./agent_demo/) is the main part of the project. It builds up agent concepts in four parts:
   1. LLM tool calling in [`1_tool_calling.py`](./agent_demo/1_tool_calling.py).
   2. An agent implemented in Python in [`2_agent.py`](./agent_demo/2_agent.py).
   3. An agent built with LangChain in [`3_langchain_agent.py`](./agent_demo/3_langchain_agent.py).
   4. Adding LangGraph to create a workflow in [`4_langgraph_workflow.py`](./agent_demo/4_langgraph_workflow.py).

## Install dependencies

This project requires Python 3.12 or newer and uses `pyproject.toml` with `uv.lock` to manage dependencies.

From the project root, run:

```bash
uv sync
```

`uv sync` creates a project virtual environment in `.venv` if needed and installs the locked dependencies. To run a command in that environment, use `uv run`, for example:

```bash
uv run python --version
```
