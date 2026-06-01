# AI Shopping Assistant - Project Pipeline & Architecture

This document provides a comprehensive overview of how the "AI Shopping Assistant" project works from end to end, explains the purpose of each file in the codebase, and details the tools and technologies used.

---

## 1. Overall Project Pipeline (How it Works)

The pipeline of the AI Shopping Assistant can be broken down into the following real-time workflow when a user interacts with the application:

### Step 1: Application Initialization & Data Loading
When the FastAPI backend server starts, it immediately loads the pre-processed e-commerce dataset (which contains products from Amazon, Flipkart, Myntra, along with their pre-calculated sentiment and value scores) into memory using Pandas. This ensures ultra-fast querying during runtime.

### Step 2: User Search Request
The user accesses the React frontend and enters a search query (e.g., "wireless headphones"). They can also adjust filters such as maximum price, minimum rating, and preferred platforms. The frontend sends this request as a JSON payload to the backend's `/search` API endpoint.

### Step 3: Initial Keyword Matching & Filtering
The backend (`main.py`) receives the request and performs an initial sweep across the loaded dataset. It filters out products that do not match the user's hard constraints (price, rating, platform). It then calculates a basic `match_score` based on how well the user's keywords match the product's name, category, and description.

### Step 4: AI Heuristic Ranking (Informed Search)
The filtered list of products is passed to the **Best First Search** algorithm (`algorithms.py`). This AI search algorithm ranks all candidates based on a composite **Heuristic Score**, which evaluates:
- **Value Score (35%)**: Price-to-rating ratio.
- **Sentiment Score (30%)**: Positive/negative sentiment of customer reviews.
- **Match Score (20%)**: Keyword relevance.
- **Normalized Rating (15%)**: Overall out-of-5 star rating.

### Step 5: LLM Augmentation
Once the products are ranked, the top product is designated as the "Best Pick". The backend then calls the **LLM Service** (`llm_service.py`), which uses Pollinations.ai to dynamically generate:
- A custom 2-sentence explanation of *why* this product is the best recommendation.
- A concise summary of thousands of customer reviews.
- An overall AI search summary for the entire query.

### Step 6: Frontend Rendering
The backend returns the ranked list of products, the Best Pick details, and the AI-generated text to the frontend. React dynamically updates the DOM, displaying the modern, glassmorphic UI with the AI summary at the top, followed by the highly detailed Best Pick card, and finally the grid of recommended products.

---

## 2. File Explanations & Project Structure

The project is divided into two main directories: `backend` for the server and AI logic, and `frontend-app` for the user interface.

### Backend (`/backend`)
*   **`main.py`**: The entry point of the FastAPI application. It defines the API endpoints (`/search`, `/filter`, `/compare`), loads the dataset on startup, handles initial keyword matching, and orchestrates the flow between algorithms and LLM services.
*   **`algorithms.py`**: The core AI logic file containing the search algorithms taught in the academic curriculum:
    *   *Breadth-First Search (BFS)*: Used for hierarchical category traversal.
    *   *Best First Search*: Ranks products using the heuristic evaluation function.
    *   *Hill Climbing*: A local search optimization algorithm used to quickly refine and filter the already-ranked products against new constraints.
*   **`llm_service.py`**: Handles all interactions with Large Language Models. It constructs prompts based on product data and sends them to the free Pollinations.ai API to generate human-readable explanations, summaries, and product comparisons.
*   **`models.py`**: Defines Pydantic data models (e.g., `SearchQuery`, `SearchResult`) to validate and type-check all incoming API requests and outgoing responses, ensuring data integrity.
*   **`config.py`**: Stores global configuration variables, dataset file paths, default limits, and centralized logging setups.
*   **Data Preparation Scripts (`scraper.py`, `train_all_datasets.py`, `train_models.py`, `nlp_engine.py`, `ml_recommender.py`)**: These scripts operate offline (before the server starts) to scrape data, perform NLP (like TF-IDF keyword extraction and VADER sentiment analysis on reviews), and train or format the dataset into the clean CSV used by the application.

### Frontend (`/frontend-app`)
*   **`src/App.jsx`**: The main React component that handles the entire user interface. It manages state (search queries, filters, loading status, product lists), sends API `fetch` requests to the backend, and maps the returned JSON data into beautiful UI components (like the Best Pick card and Product Grid).
*   **`src/index.css`**: Contains all styling for the application. It utilizes modern "glassmorphism" design principles (translucency, blur effects), CSS variables for theming, responsive grid layouts, and advanced keyframe animations (like glowing background orbs and hover micro-interactions).
*   **`vite.config.js`**: Configuration file for the Vite build tool. Notably, it sets up a proxy server so that any API calls made to `/search` or `/filter` in the frontend are automatically routed to the FastAPI backend running on `http://127.0.0.1:8000`, solving CORS issues during development.

---

## 3. Tools and Technologies Used

### Frontend Tools
*   **React (v19)**: A JavaScript library for building user interfaces. Used to create a dynamic, single-page application (SPA) without page reloads.
*   **Vite**: A blazing-fast build tool and development server that significantly speeds up the React development process compared to older tools like Create React App.
*   **Vanilla CSS**: Used instead of a framework (like Tailwind or Bootstrap) to allow for completely custom, highly-performant, and intricate visual designs like glassmorphism and dynamic background animations.

### Backend Tools
*   **Python (3.10+)**: The primary programming language for the backend, chosen for its vast AI, ML, and data processing ecosystems.
*   **FastAPI**: A modern, extremely fast web framework for building APIs with Python. Used to create the server endpoints that the frontend communicates with.
*   **Uvicorn**: An ASGI web server implementation for Python, used to serve the FastAPI application.
*   **Pandas**: A powerful data manipulation library. Used to quickly load, filter, and process the large CSV datasets of e-commerce products in memory.
*   **Pydantic**: Data validation and settings management using Python type annotations. Used within FastAPI to ensure API requests are formatted correctly.

### AI & Machine Learning Tools
*   **Pollinations.ai**: A free, zero-configuration Large Language Model API used to generate the conversational text, explanations, and review summaries without requiring API keys like OpenAI.
*   **Traditional AI Algorithms**: Best First Search, Hill Climbing, and Breadth-First Search are implemented from scratch in pure Python to demonstrate classical AI principles applied to modern ranking problems.
*   **NLP Techniques**: TF-IDF (Term Frequency-Inverse Document Frequency) for keyword matching and VADER for sentiment analysis of text reviews (utilized during the data preparation phase).
