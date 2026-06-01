# 🔗 File Interconnection Map & Dependencies

## Complete Dependency Graph

```
┌─ EXTERNAL LIBRARIES (npm, pip)
│  ├─ React, Vite, CSS
│  ├─ FastAPI, Uvicorn
│  ├─ Pandas, Scikit-learn, NLTK
│  └─ Requests, Pydantic
│
├─ FRONTEND LAYER
│  ├─ index.html
│  │  └─ main.jsx
│  │     └─ App.jsx
│  │        └─ index.css
│  └─ vite.config.js (proxy setup)
│
├─ BACKEND LAYER
│  ├─ main.py (entry point)
│  │  ├─ imports models.py
│  │  ├─ imports algorithms.py
│  │  ├─ imports llm_service.py
│  │  ├─ imports config.py
│  │  ├─ imports ml_recommender.py
│  │  └─ uses trained_models/processed_data_combined.csv
│  │
│  ├─ models.py (standalone data structures)
│  │  ├─ Product
│  │  ├─ SearchQuery
│  │  ├─ FilterRequest
│  │  └─ SearchResult
│  │
│  ├─ algorithms.py (pure functions)
│  │  ├─ bfs_category_search()
│  │  ├─ best_first_search()
│  │  ├─ hill_climbing_filter()
│  │  └─ normalize_scores()
│  │
│  ├─ llm_service.py (external API)
│  │  ├─ query_free_ai()
│  │  ├─ explain_recommendation()
│  │  ├─ summarize_reviews()
│  │  └─ calls https://text.pollinations.ai/
│  │
│  ├─ config.py (constants)
│  │  ├─ DATA_PATH → trained_models/processed_data_combined.csv
│  │  ├─ TFIDF_PATH → trained_models/tfidf_vectorizer.pkl
│  │  ├─ SCALER_PATH → trained_models/scaler.pkl
│  │  └─ DEFAULT_LIMIT = 20
│  │
│  ├─ nlp_engine.py (NLP utilities)
│  │  ├─ TFIDFEngine (for training)
│  │  ├─ SentimentAnalyzer (for training)
│  │  └─ NLPPipeline (orchestrator)
│  │
│  └─ ml_recommender.py (wrapper class)
│     ├─ imports algorithms.py
│     ├─ imports models.py
│     ├─ compute_match_scores()
│     └─ get_recommendations()
│
├─ DATA LAYER
│  ├─ Raw Data (input)
│  │  ├─ dataset/amazon.csv
│  │  ├─ Flipkart data (optional)
│  │  └─ Myntra data (optional)
│  │
│  └─ Processed Data (output, used at runtime)
│     ├─ trained_models/processed_data_combined.csv
│     ├─ trained_models/tfidf_vectorizer.pkl
│     ├─ trained_models/scaler.pkl
│     └─ trained_models/training_stats.json
│
└─ TRAINING LAYER (offline, before server starts)
   ├─ train_all_datasets.py (master)
   │  ├─ imports scraper.py
   │  ├─ imports nlp_engine.py
   │  ├─ imports algorithms.py
   │  └─ calls train_models.py functions
   │
   ├─ train_models.py (individual functions)
   └─ scraper.py (data collection)
```

---

## Function Call Chain - User Search Journey

```
┌─────────────────────────────────────────────────────────────┐
│ FRONTEND: App.jsx                                           │
│ handleSearch() {                                            │
│   const body = {                                            │
│     query: "wireless headphones",                           │
│     max_price: 5000,                                        │
│     min_rating: 4.0,                                        │
│     platforms: ["Amazon"]                                   │
│   }                                                         │
│   fetch('/search', { method: 'POST', body: JSON })         │
│ }                                                           │
└──────────────────┬────────────────────────────────────────┘
                   │ (via vite proxy)
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ BACKEND: main.py                                            │
│ @app.post("/search")                                        │
│ async def search(req: Request):                             │
│   query_data = await req.json()                             │
│   ──→ models.SearchQuery(**query_data)                      │
│        └─ Pydantic validation (type check, range check)     │
│   ──→ config.DATA_PATH                                      │
│        └─ Load: trained_models/processed_data_combined.csv  │
│   ──→ df = pd.read_csv(DATA_PATH)                           │
│        └─ Pandas DataFrame with 2000 products              │
│   ──→ df filter: price, rating, platform                    │
│        └─ 150 products match                                │
└─────────────┬───────────────────────────────────────────┬──┘
              │                                           │
              ↓                                           │
    ┌────────────────────────────────┐                   │
    │ Compute match_scores           │                   │
    │ for each product:              │                   │
    │                                │                   │
    │ tfidf_vectorizer.pkl loaded    │                   │
    │ ──→ cosine_similarity(         │                   │
    │     query, product_name        │                   │
    │   ) ∈ [0, 1]                   │                   │
    │                                │                   │
    │ products['match_score'] = ...  │                   │
    └────────────┬────────────────────┘                   │
                 │                                        │
                 ↓                                        │
    ┌────────────────────────────────┐                   │
    │ algorithms.normalize_scores()  │                   │
    │                                │                   │
    │ For each product:              │                   │
    │   value_score = (              │                   │
    │     rating_norm × log(reviews) │                   │
    │   ) / (price_norm + ε)         │                   │
    │                                │                   │
    │ All scores → [0, 1]            │                   │
    └────────────┬────────────────────┘                   │
                 │                                        │
                 ↓                                        │
    ┌────────────────────────────────┐                   │
    │ algorithms.best_first_search() │                   │
    │ (150 products)                 │                   │
    │                                │                   │
    │ For each product:              │                   │
    │   h(n) = 0.35×value +          │                   │
    │           0.30×sentiment +     │                   │
    │           0.20×match +         │                   │
    │           0.15×rating_norm     │                   │
    │                                │                   │
    │ nlargest(10, by=h_score)       │                   │
    │                                │                   │
    │ Result: top 10 products        │                   │
    │ ranked_products[0] = best_pick │                   │
    └────────────┬────────────────────┘                   │
                 │                                        │
                 ↓                                        │
    ┌────────────────────────────────┐                   │
    │ llm_service.                   │                   │
    │ explain_recommendation()       │                   │
    │ (best_product, query)          │                   │
    │                                │                   │
    │ prompt = f"""...               │                   │
    │   {best_product.name}          │                   │
    │   {best_product.price}         │                   │
    │   {best_product.rating}        │                   │
    │   {query}                      │                   │
    │ """                            │                   │
    │                                │                   │
    │ ──→ query_free_ai(prompt)      │                   │
    │     └─ POST https://...        │                   │
    │     .pollinations.ai/          │                   │
    │     └─ Response: "This is best │                   │
    │        because..."             │                   │
    │                                │                   │
    │ best_product['ai_explanation'] │                   │
    │   = response_text              │                   │
    └────────────┬────────────────────┘                   │
                 │                                        ↓
                 │     ┌──────────────────────────────────┐
                 │     │ Generate ai_summary for          │
                 │     │ all results                      │
                 │     │                                  │
                 │     │ Orchestrate top 3 products +    │
                 │     │ query to create summary         │
                 │     └──────────────────┬───────────────┘
                 │                        │
                 ↓                        ↓
    ┌──────────────────────────────────────────┐
    │ models.SearchResult validation           │
    │                                          │
    │ SearchResult(                            │
    │   products=[10 Product objects],         │
    │   best_pick=products[0],                 │
    │   ai_summary="Found 150 products...",    │
    │   total_found=150                        │
    │ )                                        │
    │ ──→ Pydantic validation                  │
    │ ──→ Convert to JSON                      │
    │ ──→ Return to frontend                   │
    └──────────────────┬───────────────────────┘
                       │
                       ↓ (JSON response)
┌─────────────────────────────────────────────────────────────┐
│ FRONTEND: App.jsx (handleSearch continued)                  │
│ const data = await res.json()                               │
│                                                             │
│ setProducts(data.products)       // 10 products             │
│ setBestPick(data.best_pick)      // #1 product              │
│ setAiSummary(data.ai_summary)    // AI text                 │
│                                                             │
│ Re-render with state                                        │
│ └─ Show AI Summary card                                     │
│ └─ Show Best Pick hero card                                 │
│ └─ Show Product grid                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## File Import/Export Matrix

```
┌──────────────────────────────────────────────────────────────────┐
│ FROM FILE         │ IMPORTS              │ EXPORTS / USED BY    │
├──────────────────────────────────────────────────────────────────┤
│ main.py          │ • models.py          │ Used by: Frontend    │
│                  │ • algorithms.py      │ /search endpoint     │
│                  │ • llm_service.py     │ Returns SearchResult │
│                  │ • config.py          │                      │
│                  │ • pandas              │                      │
│                  │ • fastapi            │                      │
├──────────────────────────────────────────────────────────────────┤
│ algorithms.py    │ • numpy              │ Used by: main.py     │
│                  │ • heapq              │ Returns: ranked list │
│                  │ • typing             │ of products          │
│                  │ • collections        │                      │
│                  │ (PURE PYTHON)        │                      │
├──────────────────────────────────────────────────────────────────┤
│ models.py        │ • pydantic           │ Used by: main.py     │
│                  │ • typing             │ Product, SearchQuery,│
│                  │ (PURE DATA)          │ SearchResult         │
├──────────────────────────────────────────────────────────────────┤
│ llm_service.py   │ • requests           │ Used by: main.py     │
│                  │ • logging            │ explain_recomm...()  │
│                  │ • typing             │ summarize_reviews()  │
│                  │ • (external API)     │ search_summary()     │
├──────────────────────────────────────────────────────────────────┤
│ config.py        │ • logging            │ Used by: main.py     │
│                  │ • pathlib            │ DATA_PATH, logger    │
│                  │ (PURE CONFIG)        │ TFIDF_PATH, etc      │
├──────────────────────────────────────────────────────────────────┤
│ nlp_engine.py    │ • sklearn (TF-IDF)   │ Used by:             │
│                  │ • nltk (VADER)       │ train_all_datasets   │
│                  │ • numpy              │ (offline only)       │
│                  │ • typing             │                      │
├──────────────────────────────────────────────────────────────────┤
│ ml_recommender.py│ • algorithms.py      │ Used by: tests       │
│                  │ • models.py          │ get_recommendations()│
│                  │ • pandas             │ compute_match...()  │
│                  │ • joblib             │                      │
├──────────────────────────────────────────────────────────────────┤
│ train_all_       │ • nlp_engine.py      │ Creates:             │
│ datasets.py      │ • algorithms.py      │ processed_data.csv   │
│                  │ • scraper.py         │ .pkl files           │
│                  │ • train_models.py    │ .json stats          │
│                  │ • pandas             │                      │
│                  │ • sklearn            │                      │
├──────────────────────────────────────────────────────────────────┤
│ App.jsx          │ • React              │ Frontend UI          │
│                  │ • fetch API (HTTP)   │ Calls: /search       │
│                  │ • index.css          │        /filter       │
│                  │ • vite proxy         │        /compare      │
├──────────────────────────────────────────────────────────────────┤
│ index.css        │ • CSS3               │ Used by: App.jsx     │
│                  │ • (no imports)       │ Styling             │
├──────────────────────────────────────────────────────────────────┤
│ vite.config.js   │ • @vitejs/...        │ Routes /search → :800│
│                  │ • defineConfig       │ Routes /filter → :800│
│                  │ • (no business logic)│ Routes /compare → 800│
└──────────────────────────────────────────────────────────────────┘
```

---

## Data Structure Relationships

```
PYDANTIC MODELS (models.py):
┌────────────────────────────────────────────────────────┐
│ Product                                                │
├────────────────────────────────────────────────────────┤
│ • product_name: str                                    │
│ • price: float                                         │
│ • rating: float [0-5]                                  │
│ • review_count: int                                    │
│ • platform: str (Amazon | Flipkart | Myntra)         │
│ • url, image_url: str                                  │
│ • sentiment_score: float [-1 to +1]  ← VADER          │
│ • value_score: float [0-1]  ← value calculation       │
│ • match_score: float [0-1]  ← TF-IDF                  │
│ • heuristic_score: float [0-1]  ← Best First          │
│ • ai_explanation: str  ← LLM generated                │
└────────────────────────────────────────────────────────┘
                        ↕
                   Used by many

┌────────────────────────────────────────────────────────┐
│ SearchQuery (User's request)                           │
├────────────────────────────────────────────────────────┤
│ • query: str                  ("wireless headphones")  │
│ • max_price: float            (5000)                   │
│ • min_rating: float [0-5]     (4.0)                    │
│ • platforms: List[str]        (["Amazon"])            │
│ • limit: int [1-100]          (10)                     │
└────────────────────────────────────────────────────────┘
                        ↓
             main.py:/search endpoint

┌────────────────────────────────────────────────────────┐
│ SearchResult (API response)                            │
├────────────────────────────────────────────────────────┤
│ • products: List[Product]     (10 ranked)              │
│ • best_pick: Product          (top 1)                  │
│ • ai_summary: str             ("Found 150...")         │
│ • total_found: int            (150)                    │
└────────────────────────────────────────────────────────┘
                        ↓
              Frontend receives JSON
              App.jsx:handleSearch
```

---

## Scoring Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ RAW PRODUCT DATA (from CSV)                                     │
│ ├─ product_name: "Sony WH-1000XM5"                              │
│ ├─ price: 24990                                                 │
│ ├─ rating: 4.7                                                  │
│ ├─ review_count: 12400                                          │
│ └─ reviews: ["Great sound", "Best buy", ...]                   │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
        ┌──────────────────────────────────────┐
        │ OFFLINE (train_all_datasets.py)      │
        │                                      │
        │ 1. VADER Sentiment Analysis          │
        │    nlp_engine.SentimentAnalyzer()    │
        │    └─ sentiment_score ∈ [-1, +1]    │
        │       (average of review sentiments)│
        │                                      │
        │ 2. Normalize & Compute Value         │
        │    algorithms.normalize_scores()     │
        │    ├─ price_norm = (p - min) / range│
        │    ├─ rating_norm = rating / 5      │
        │    ├─ value_score = (rating_norm ×  │
        │    │   log(review_count+1)) /       │
        │    │   (price_norm + 0.01)           │
        │    └─ value_score ∈ [0, 1]          │
        │                                      │
        │ 3. Save TF-IDF Vectorizer           │
        │    ├─ tfidf_vectorizer.pkl          │
        │    └─ Used for match_score later    │
        └──────────────────┬───────────────────┘
                           ↓
    ┌──────────────────────────────────────┐
    │ SAVED: processed_data_combined.csv   │
    │ (All 2000 products with scores)      │
    │ ├─ product_name                      │
    │ ├─ price                             │
    │ ├─ rating                            │
    │ ├─ sentiment_score  ← CALCULATED     │
    │ └─ value_score      ← CALCULATED     │
    └──────────────┬───────────────────────┘
                   ↓
        ┌──────────────────────────────────────┐
        │ RUNTIME (main.py:/search)            │
        │                                      │
        │ 4. Compute Match Score               │
        │    Using TF-IDF vectorizer.pkl       │
        │    ├─ Vectorize query & product_name│
        │    ├─ cosine_similarity()            │
        │    └─ match_score ∈ [0, 1]          │
        │                                      │
        │ 5. Compute Heuristic Score          │
        │    algorithms.best_first_search()    │
        │    h(n) = 0.35×value +              │
        │            0.30×sentiment +         │
        │            0.20×match +             │
        │            0.15×rating_norm         │
        │    └─ heuristic_score ∈ [0, 1]      │
        │                                      │
        │ 6. Rank by Heuristic                │
        │    nlargest(top_n, products)        │
        │    └─ products[0] = Best Pick       │
        └──────────────┬───────────────────────┘
                       ↓
    ┌──────────────────────────────────────┐
    │ FINAL PRODUCT (with all scores)      │
    │ ├─ product_name: "Sony..."           │
    │ ├─ price: 24990                      │
    │ ├─ rating: 4.7                       │
    │ ├─ sentiment_score: 0.85             │
    │ ├─ value_score: 0.78                 │
    │ ├─ match_score: 0.92                 │
    │ ├─ heuristic_score: 0.87             │
    │ └─ ai_explanation: "Best choice..." │
    └──────────────────────────────────────┘
```

---

## API Endpoint Flow

```
ENDPOINT: POST /search
┌────────────────────────────────────────────────────────┐
│ INPUT:  SearchQuery (JSON)                             │
│ {                                                      │
│   "query": "wireless headphones",                      │
│   "max_price": 5000,                                   │
│   "min_rating": 4.0,                                   │
│   "platforms": ["Amazon", "Flipkart"],                 │
│   "limit": 10                                          │
│ }                                                      │
└────────────────────────────────────────────────────────┘
                      │
                      ↓
        ┌─────────────────────────────┐
        │ Validation (Pydantic)        │
        │ ✓ query: str                 │
        │ ✓ max_price: float ≥ 0      │
        │ ✓ min_rating: 0 ≤ x ≤ 5     │
        │ ✓ platforms: known values    │
        │ ✓ limit: 1 ≤ x ≤ 100         │
        └─────────────────────────────┘
                      │
                      ↓
        ┌─────────────────────────────┐
        │ Processing (main.py)        │
        │ ├─ Load CSV → DataFrame     │
        │ ├─ Filter by constraints    │
        │ ├─ Compute match_scores     │
        │ ├─ Normalize scores         │
        │ ├─ Best First Search        │
        │ ├─ LLM Enhancement          │
        │ └─ Build response           │
        └─────────────────────────────┘
                      │
                      ↓
┌────────────────────────────────────────────────────────┐
│ OUTPUT: SearchResult (JSON)                            │
│ {                                                      │
│   "products": [                                        │
│     {                                                  │
│       "product_name": "Sony WH-1000XM5",              │
│       "price": 24990,                                  │
│       "rating": 4.7,                                   │
│       "sentiment_score": 0.85,                         │
│       "match_score": 0.92,                             │
│       "value_score": 0.78,                             │
│       "heuristic_score": 0.87,                         │
│       "ai_explanation": "Best choice..."              │
│     },                                                 │
│     ... (9 more)                                       │
│   ],                                                   │
│   "best_pick": { ... },                               │
│   "ai_summary": "We found 150 products...",           │
│   "total_found": 150                                   │
│ }                                                      │
└────────────────────────────────────────────────────────┘
```

---

## Environment & Deployment Context

```
┌──────────────────────────────────────────────────────────┐
│ DEVELOPMENT ENVIRONMENT                                  │
├──────────────────────────────────────────────────────────┤
│ • OS: Windows / Linux / macOS                            │
│ • Python: 3.10+                                          │
│ • Node: 16+                                              │
│                                                          │
│ FOLDER STRUCTURE:                                        │
│ AI Shopping Assistant/                                   │
│ ├─ backend/                                              │
│ │  ├─ main.py                                            │
│ │  ├─ algorithms.py                                      │
│ │  ├─ models.py                                          │
│ │  ├─ llm_service.py                                     │
│ │  ├─ nlp_engine.py                                      │
│ │  ├─ config.py                                          │
│ │  ├─ ml_recommender.py                                  │
│ │  ├─ trained_models/                                    │
│ │  │  ├─ processed_data_combined.csv                     │
│ │  │  ├─ tfidf_vectorizer.pkl                            │
│ │  │  ├─ scaler.pkl                                      │
│ │  │  └─ training_stats.json                             │
│ │  ├─ requirements.txt                                   │
│ │  └─ .env (API keys, if needed)                         │
│ │                                                        │
│ ├─ frontend-app/                                         │
│ │  ├─ src/                                               │
│ │  │  ├─ App.jsx                                         │
│ │  │  ├─ main.jsx                                        │
│ │  │  └─ index.css                                       │
│ │  ├─ vite.config.js                                     │
│ │  ├─ index.html                                         │
│ │  ├─ package.json                                       │
│ │  └─ .env (if needed)                                   │
│ │                                                        │
│ ├─ dataset/                                              │
│ │  ├─ amazon.csv/                                        │
│ │  │  └─ amazon.csv                                      │
│ │  ├─ flipkart.csv (optional)                           │
│ │  └─ myntra.csv (optional)                             │
│ │                                                        │
│ ├─ training/                                             │
│ │  ├─ train_models.py                                    │
│ │  └─ train_all_datasets.py                              │
│ │                                                        │
│ └─ tests/                                                │
│    ├─ test_api.py                                        │
│    ├─ test_ml_complete.py                                │
│    └─ ... (other test files)                             │
└──────────────────────────────────────────────────────────┘

RUNTIME FLOW:
1. Backend startup
   └─ config.py sets DATA_PATH
   └─ main.py loads CSV into memory
   └─ Uvicorn server on :8000

2. Frontend startup
   └─ Vite dev server on :5173
   └─ vite.config.js configures proxy

3. User interaction
   └─ App.jsx:handleSearch()
   └─ fetch POST to /search
   └─ Backend processes
   └─ Returns JSON
   └─ Frontend renders UI
```

---

## Module Responsibilities

```
┌────────────────────────────────────────────────────────┐
│ main.py                                                │
├────────────────────────────────────────────────────────┤
│ RESPONSIBILITY: Orchestrate entire search pipeline     │
│                                                        │
│ Key Functions:                                         │
│ • @app.post("/search") → Main endpoint                 │
│ • @app.get("/") → Health check                         │
│ • Load & cache processed_data_combined.csv             │
│ • Call algorithms, LLM, format response                │
│                                                        │
│ Decision Points:                                       │
│ • Which products to filter?                            │
│ • Call which algorithm? (Best First Search)            │
│ • Generate AI explanation? (LLM)                       │
│ • Return best_pick or error?                           │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ algorithms.py                                          │
├────────────────────────────────────────────────────────┤
│ RESPONSIBILITY: Implement AI search algorithms         │
│                                                        │
│ Functions:                                             │
│ • bfs_category_search() → Category traversal           │
│ • best_first_search() → Ranking by heuristic          │
│ • hill_climbing_filter() → Local optimization          │
│ • normalize_scores() → Min-max normalization           │
│                                                        │
│ Pure Functions (no side effects):                      │
│ • Take data in, return results out                     │
│ • No external API calls                                │
│ • No file I/O                                          │
│ • No state changes                                     │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ models.py                                              │
├────────────────────────────────────────────────────────┤
│ RESPONSIBILITY: Define & validate data structures      │
│                                                        │
│ Pydantic Models:                                       │
│ • Product → Represents one product                     │
│ • SearchQuery → Represents user request                │
│ • SearchResult → Represents API response               │
│ • FilterRequest → Request to Hill Climbing             │
│                                                        │
│ Validation Included:                                   │
│ • Type checking (str, float, int, List)                │
│ • Range checking (0-5 ratings, 0-1 scores)            │
│ • Enum checking (platforms in known list)              │
│ • Required field checking (no missing fields)          │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ llm_service.py                                         │
├────────────────────────────────────────────────────────┤
│ RESPONSIBILITY: Generate AI text via LLM               │
│                                                        │
│ External Dependency:                                   │
│ • Pollinations.ai (free, no key)                       │
│ • HTTP timeout: 30 seconds                             │
│ • Fallback text if API fails                           │
│                                                        │
│ Functions:                                             │
│ • query_free_ai(prompt) → Raw LLM response             │
│ • explain_recommendation() → 2-sentence explanation    │
│ • summarize_reviews() → Review summary                 │
│ • search_summary() → Overall summary                   │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ nlp_engine.py                                          │
├────────────────────────────────────────────────────────┤
│ RESPONSIBILITY: NLP utilities for training & inference │
│                                                        │
│ Used During Training:                                  │
│ • TFIDFEngine.fit() → Learn vocabulary                 │
│ • Save vectorizer to .pkl                              │
│                                                        │
│ Used During Inference (main.py):                       │
│ • Load tfidf_vectorizer.pkl                            │
│ • Compute cosine_similarity for match_score            │
│                                                        │
│ Also Includes:                                         │
│ • SentimentAnalyzer (VADER)                            │
│ • NLPPipeline orchestrator                             │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ config.py                                              │
├────────────────────────────────────────────────────────┤
│ RESPONSIBILITY: Centralized configuration              │
│                                                        │
│ Provides:                                              │
│ • DATA_PATH → CSV file location                        │
│ • TFIDF_PATH → Vectorizer file location                │
│ • SCALER_PATH → Scaler file location                   │
│ • DEFAULT_LIMIT → Default result count (20)            │
│ • logger → Logging setup                               │
│                                                        │
│ Why Centralize?                                        │
│ • Easy to change for different environments            │
│ • Single source of truth                               │
│ • Used by multiple modules                             │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ ml_recommender.py                                      │
├────────────────────────────────────────────────────────┤
│ RESPONSIBILITY: Lightweight wrapper for testing        │
│                                                        │
│ Provides:                                              │
│ • Recommender class                                    │
│ • compute_match_scores()                               │
│ • get_recommendations() → Full pipeline                │
│                                                        │
│ Used By:                                               │
│ • Test files (test_api.py, test_ml_complete.py)       │
│ • Not used by main.py (main.py does this directly)    │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ App.jsx                                                │
├────────────────────────────────────────────────────────┤
│ RESPONSIBILITY: React UI & state management            │
│                                                        │
│ State Variables:                                       │
│ • query, maxPrice, minRating                           │
│ • platforms, loading, products                         │
│ • bestPick, aiSummary, error                           │
│                                                        │
│ Key Functions:                                         │
│ • handleSearch() → API call                            │
│ • handlePlatformChange() → Update filters              │
│ • getSortedProducts() → Client-side sort               │
│                                                        │
│ Renders:                                               │
│ • Search input & filters                               │
│ • AI Summary card                                      │
│ • Best Pick hero card                                  │
│ • Product grid                                         │
│ • Error messages                                       │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ vite.config.js                                         │
├────────────────────────────────────────────────────────┤
│ RESPONSIBILITY: Build tool & dev server configuration  │
│                                                        │
│ Configures:                                            │
│ • React plugin for JSX                                 │
│ • Dev server on localhost:5173                         │
│ • Proxy: /search → http://127.0.0.1:8000              │
│ • Proxy: /filter → http://127.0.0.1:8000              │
│ • Proxy: /compare → http://127.0.0.1:8000             │
│                                                        │
│ Why Proxy?                                             │
│ • Solves CORS issues during development                │
│ • Allows cross-origin requests                         │
│ • Frontend thinks backend is same origin               │
└────────────────────────────────────────────────────────┘
```

---

## Request/Response Examples

```
REQUEST EXAMPLE:
┌────────────────────────────────────────────────────────┐
│ POST http://localhost:5173/search                       │
│                                                        │
│ Headers:                                               │
│ Content-Type: application/json                         │
│                                                        │
│ Body:                                                  │
│ {                                                      │
│   "query": "boAt earphones bass",                      │
│   "max_price": 2000,                                   │
│   "min_rating": 3.5,                                   │
│   "platforms": ["Amazon", "Flipkart"],                 │
│   "limit": 10                                          │
│ }                                                      │
└────────────────────────────────────────────────────────┘

RESPONSE EXAMPLE:
┌────────────────────────────────────────────────────────┐
│ HTTP 200 OK                                            │
│                                                        │
│ Body:                                                  │
│ {                                                      │
│   "products": [                                        │
│     {                                                  │
│       "product_id": "boat-551",                        │
│       "product_name": "boAt Rockerz 551",              │
│       "price": 1299,                                   │
│       "discounted_price": 1299,                        │
│       "rating": 4.2,                                   │
│       "rating_count": 3621,                            │
│       "platform": "Amazon",                            │
│       "url": "https://amazon.in/...",                 │
│       "image_url": "https://images/...",              │
│       "reviews": ["Great bass", "Value for money"],   │
│       "category": "Audio",                             │
│       "sentiment_score": 0.75,                         │
│       "value_score": 0.82,                             │
│       "match_score": 0.87,                             │
│       "heuristic_score": 0.812,                        │
│       "ai_explanation": "boAt Rockerz 551 delivers..."│
│     },                                                 │
│     { ... }, (9 more products)                         │
│   ],                                                   │
│   "best_pick": { ... },  // Same as products[0]        │
│   "ai_summary": "We found 45 products...",             │
│   "total_found": 45                                    │
│ }                                                      │
└────────────────────────────────────────────────────────┘

ERROR RESPONSE (Validation Failure):
┌────────────────────────────────────────────────────────┐
│ HTTP 422 Unprocessable Entity                          │
│                                                        │
│ Body:                                                  │
│ {                                                      │
│   "detail": [                                          │
│     {                                                  │
│       "loc": ["body", "min_rating"],                   │
│       "msg": "value is not a valid float",             │
│       "type": "type_error.float"                       │
│     }                                                  │
│   ]                                                    │
│ }                                                      │
└────────────────────────────────────────────────────────┘
```

---

This comprehensive interconnection map shows exactly how every file relates to every other file and what flows between them!

