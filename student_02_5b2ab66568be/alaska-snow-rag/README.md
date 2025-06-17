# Alaska FAQ RAG System

A Retrieval-Augmented Generation (RAG) system for answering questions about Alaska Department services using BigQuery vector search and Google's Gemini AI.

## Features

- **RAG System**: Searches Alaska FAQ knowledge base using vector similarity
- **Prompt Validation**: Ensures user prompts are safe before processing
- **Response Generation**: Uses Gemini AI to generate contextual answers
- **Unit Testing**: Comprehensive tests with mocked dependencies
- **Quality Evaluation**: Uses Google Evaluation Service to assess response quality

#run local RAG
python test_local.py

# Run all unit tests
pytest test_unit.py -v

# Run specific test class
pytest test_unit.py::TestRAGSystem -v

# Run with coverage
pytest test_unit.py --cov=. --cov-report=html

# Full evaluation with Google Evaluation Service
python evaluation.py

# Quick local evaluation (without Google service)
python evaluation.py --quick