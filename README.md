# ShopMind: AI-Powered Shopping Assistant 

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)

An intelligent, full-stack e-commerce shopping assistant. ShopMind aggregates product data across multiple platforms and uses a hybrid of traditional **AI Search Algorithms** (Best-First, Hill Climbing) and modern **Large Language Models (LLMs)** to dynamically rank products and generate automated buy-recommendations.

## Core Features

###  Multi-Platform Aggregation
Instantly queries, aggregates, and filters e-commerce datasets across multiple top platforms simultaneously, providing a unified shopping experience.

###  Hybrid AI Ranking Engine
Employs traditional AI search algorithms (Best-First Search, Hill Climbing) combined with NLP sentiment analysis (VADER) and TF-IDF keyword matching to score products based on price-to-value ratios and customer sentiment.

### LLM-Powered Explanations
Integrates with the OpenAI API to read thousands of customer reviews and generate instant, human-readable explanations of exactly why a product is the "Best Pick."

###  Modern UI/UX
Features a beautifully crafted, highly responsive frontend utilizing React, Vite, and modern glassmorphism aesthetics for dynamic side-by-side product comparisons.

##  Technology Stack
*   **Backend:** Python 3.10+, FastAPI, Uvicorn, Pandas
*   **Frontend:** React 19, Vite, Vanilla CSS
*   **AI / Machine Learning:** OpenAI API, NLTK (VADER Sentiment Analysis), TF-IDF, Heuristic Search Algorithms

## Getting Started

### 1. Start the Backend
```bash
cd backend
pip install -r requirements.txt
# Create a .env file and add your OPENAI_API_KEY
python main.py
