# RAG Application with Azure AI Search

Retrieval-Augmented Generation app that grounds LLM answers in private documents using vector search and semantic ranking.

## Stack

- Azure AI Search
- Azure OpenAI
- Embeddings
- Python

## What this project does

This project demonstrates a basic Retrieval-Augmented Generation architecture using Python, Azure OpenAI, and Azure AI Search.

The application reads private documents, splits them into smaller chunks, generates embeddings with Azure OpenAI, stores the chunks and vectors in Azure AI Search, and then answers user questions based on retrieved document context.

## Architecture

User question
→ Azure OpenAI embedding
→ Azure AI Search vector and hybrid search
→ Retrieved document chunks
→ Azure OpenAI chat completion
→ Grounded answer with sources

## Project structure

```text
rag-application-azure-ai-search/
│
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── data/
│   └── sample-document.txt
│
└── src/
    ├── __init__.py
    ├── config.py
    ├── chunking.py
    ├── embeddings.py
    ├── search_index.py
    ├── ingest.py
    └── ask.py
```

## Setup

Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file based on `.env.example` and fill in your Azure credentials.

```powershell
Copy-Item .env.example .env
```

## Run ingestion

```powershell
python -m src.ingest
```

## Ask a question

```powershell
python -m src.ask "What is a RAG application?"
```

## Important security note

Do not commit your `.env` file to GitHub. It contains credentials and API keys.

## Cost note

This project uses Azure resources. For labs, create all resources in a dedicated Resource Group and delete the Resource Group after testing.
