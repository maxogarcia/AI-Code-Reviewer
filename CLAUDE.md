# AI Code Reviewer

## Project Overview
LLM-powered code review assistant. QLoRA fine-tuned Qwen2.5-Coder-7B-Instruct on code review data, with a RAG pipeline for context-aware feedback. Full-stack app with FastAPI backend and React frontend.

## Stack
- Model: Qwen2.5-Coder-7B-Instruct (QLoRA fine-tuned via PEFT/TRL)
- RAG: ChromaDB + sentence-transformers
- Backend: FastAPI
- Frontend: React
- Experiment Tracking: MLflow
- Infrastructure: Docker, GitHub Actions CI/CD

## Environment
- Python 3.13, venv at .venv/
- CUDA 13.2, RTX 4080 Laptop GPU
- Activate: source .venv/bin/activate

## Dataset
- Fine-tuning: alenphilip/Code-Review-Assistant (13,670 examples, Qwen2.5 chat format)
- RAG corpus: Tomo-Melb/CodeReviewQA

## Project Structure
- training/   - QLoRA fine-tuning scripts
- rag/        - Embedding, vector DB, retrieval pipeline
- api/        - FastAPI backend
- frontend/   - React app
- data/       - Dataset scripts
- mlflow/     - Experiment tracking config
- docker/     - Dockerfiles
