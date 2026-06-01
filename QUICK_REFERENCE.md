# 🎯 AI Shopping Assistant - Quick Visual Reference

## System Architecture Overview

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                         REACT FRONTEND                              ┃
┃                      (localhost:5173)                               ┃
┃                                                                      ┃
┃  ┌──────────────────────────────────────────────────────────────┐ ┃
┃  │ App.jsx - Search Box + Filters + UI Rendering               │ ┃
┃  │ └─ State: query, products[], bestPick, aiSummary, error     │ ┃
┃  └──────────────────────────────────────────────────────────────┘ ┃
┃                              │                                      ┃
┃                              │ HTTP POST                            ┃
┃                              │ (/search endpoint)                  ┃
┃                              ↓                                      ┃
┃  ┌──────────────────────────────────────────────────────────────┐ ┃
┃  │ Vite Dev Server (Proxy)                                      │ ┃
┃  │ Routes /search → http://127.0.0.1:8000/search              │ ┃
┃  └──────────────────────────────────────────────────────────────┘ ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                              │
                              ↓
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                      FASTAPI BACKEND                                ┃
┃                   (http://127.0.0.1:8000)                          ┃
┃                                                                      ┃
┃  main.py - Central Orchestrator                                     ┃
┃  ├─ Loads: processed_data_combined.csv (Pandas DataFrame)          ┃
┃  ├─ Receives: SearchQuery (Pydantic validation)                    ┃
┃  │                                                                   ┃
┃  └─ PIPELINE:                                                       ┃
┃     │                                                               ┃
┃     ├─→ STEP 1: Filter Constraints                                 ┃
┃     │   └─ price ≤ max_price, rating ≥ min_rating                ┃
┃     │   └─ platform in ["Amazon", "Flipkart", "Myntra"]          ┃
┃     │   └─ Result: ~50-200 matching products                     ┃
┃     │                                                               ┃
┃     ├─→ STEP 2: Keyword Matching (NLP)                            ┃
┃     │   └─ TF-IDF vectorization + cosine similarity               ┃
┃     │   └─ match_score ∈ [0, 1] for each product                 ┃
┃     │                                                               ┃
┃     ├─→ STEP 3: Normalize Scores                                  ┃
┃     │   └─ algorithms.normalize_scores()                          ┃
┃     │   └─ Ensures value_score computed & clamped [0,1]         ┃
┃     │                                                               ┃
┃     ├─→ STEP 4: Best First Search (AI Ranking)                   ┃
┃     │   └─ algorithms.best_first_search()                         ┃
┃     │   └─ h(n) = 0.35×value + 0.30×sentiment +                 ┃
┃     │              0.20×match + 0.15×rating                       ┃
┃     │   └─ Sort by heuristic_score (descending)                  ┃
┃     │   └─ Top product = "Best Pick"                             ┃
┃     │                                                               ┃
┃     ├─→ STEP 5: LLM Enhancement                                   ┃
┃     │   └─ llm_service.explain_recommendation()                   ┃
┃     │   └─ Polls Pollinations.ai (free LLM)                       ┃
┃     │   └─ Generates 2-sentence AI explanation                    ┃
┃     │                                                               ┃
┃     └─→ STEP 6: Return SearchResult (JSON)                        ┃
┃         └─ models.SearchResult (Pydantic validation)              ┃
┃         └─ products[], best_pick, ai_summary, total_found         ┃
┃                                                                      ┃
┃  Supporting Modules:                                                ┃
┃  ├─ models.py      → Data validation (Pydantic)                    ┃
┃  ├─ algorithms.py  → BFS, Best First, Hill Climbing                ┃
┃  ├─ llm_service.py → Pollinations.ai integration                   ┃
┃  ├─ nlp_engine.py  → TF-IDF & VADER sentiment                      ┃
┃  ├─ config.py      → Paths & constants                             ┃
┃  └─ ml_recommender.py → Lightweight wrapper (tests)                ┃
┃                                                                      ┃
┃  Data Files:                                                        ┃
┃  └─ trained_models/                                                ┃
┃     ├─ processed_data_combined.csv     (Main dataset - 2000 products)
┃     ├─ tfidf_vectorizer.pkl           (For keyword matching)       ┃
┃     ├─ scaler.pkl                     (Feature normalization)      ┃
┃     └─ training_stats.json            (Metrics)                    ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                              │
                              │ JSON Response
                              ↓
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                      FRONTEND DISPLAY                                ┃
┃                                                                      ┃
┃  ┌──────────────────────────────────────────────────────────────┐ ┃
┃  │ AI Summary Card                                              │ ┃
┃  │ "We found 45 products... The best option is..."             │ ┃
┃  └──────────────────────────────────────────────────────────────┘ ┃
┃                                                                      ┃
┃  ┌──────────────────────────────────────────────────────────────┐ ┃
┃  │ BEST PICK Hero Card                                          │ ┃
┃  │ ┌────────┐ Product Name                                      │ ┃
┃  │ │ IMAGE  │ Price  | Rating  | Review Count                  │ ┃
┃  │ │        │ ⭐4.7  | 12,400 reviews                            │ ┃
┃  │ └────────┘ AI Explanation: "This is best because..."        │ ┃
┃  │           [View on Amazon →]                                 │ ┃
┃  └──────────────────────────────────────────────────────────────┘ ┃
┃                                                                      ┃
┃  ┌──────────────────────────────────────────────────────────────┐ ┃
┃  │ Product Grid (9 more products)                               │ ┃
┃  │ ┌─────────┐ ┌─────────┐ ┌─────────┐                         │ ┃
┃  │ │Product 2│ │Product 3│ │Product 4│ ...                    │ ┃
┃  │ └─────────┘ └─────────┘ └─────────┘                         │ ┃
┃  └──────────────────────────────────────────────────────────────┘ ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

---

## File Purpose Matrix

```
┌─────────────────────────────────────────────────────────────────────────┐
│ FILE                      │ LANGUAGE  │ PURPOSE                         │
├─────────────────────────────────────────────────────────────────────────┤
│ FRONTEND                                                                │
├─────────────────────────────────────────────────────────────────────────┤
│ App.jsx                   │ React     │ Main UI component & state mgmt  │
│ index.css                 │ CSS3      │ Glassmorphic styling            │
│ main.jsx                  │ React     │ Entry point (renders App)       │
│ index.html                │ HTML5     │ Root HTML file                  │
│ vite.config.js            │ JS        │ Build config & dev proxy        │
│ package.json              │ JSON      │ Dependencies & scripts          │
├─────────────────────────────────────────────────────────────────────────┤
│ BACKEND - CORE                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ main.py                   │ Python    │ API server & orchestrator       │
│ models.py                 │ Python    │ Pydantic data validation        │
│ algorithms.py             │ Python    │ BFS, Best First, Hill Climbing │
│ llm_service.py            │ Python    │ LLM integration (Pollinations)  │
├─────────────────────────────────────────────────────────────────────────┤
│ BACKEND - SUPPORTING                                                   │
├─────────────────────────────────────────────────────────────────────────┤
│ config.py                 │ Python    │ Paths, constants, logging       │
│ nlp_engine.py             │ Python    │ TF-IDF & VADER sentiment        │
│ ml_recommender.py         │ Python    │ Lightweight recommender class   │
│ scraper.py                │ Python    │ Data scraping (offline)         │
│ train_all_datasets.py     │ Python    │ Master training pipeline        │
│ train_models.py           │ Python    │ ML model training               │
├─────────────────────────────────────────────────────────────────────────┤
│ DATA                                                                    │
├─────────────────────────────────────────────────────────────────────────┤
│ processed_data_combined   │ CSV       │ Main dataset (2000 products)    │
│ tfidf_vectorizer.pkl      │ PKL       │ Trained TF-IDF model            │
│ scaler.pkl                │ PKL       │ Feature normalization model     │
│ training_stats.json       │ JSON      │ Training metrics                │
├─────────────────────────────────────────────────────────────────────────┤
│ DOCUMENTATION                                                           │
├─────────────────────────────────────────────────────────────────────────┤
│ README.md                 │ Markdown  │ Project overview                │
│ README_PIPELINE.md        │ Markdown  │ Pipeline explanation            │
│ ALGORITHMS_GUIDE.md       │ Markdown  │ Algorithm details               │
│ QUICK_START.md            │ Markdown  │ Setup instructions              │
│ VIVA_CHEAT_SHEET.md       │ Markdown  │ Exam preparation notes          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Search Query Example

```
USER TYPES: "wireless headphones under 5000"
            max_price: 5000
            min_rating: 4.0
            platforms: ["Amazon", "Flipkart"]
                          │
                          ↓
        ┌─────────────────────────────────────┐
        │ App.jsx: handleSearch()              │
        │ Fetch POST /search                  │
        │ Body: SearchQuery JSON              │
        └─────────────────────────────────────┘
                          │
                          ↓
        ┌─────────────────────────────────────┐
        │ Pydantic: SearchQuery validation    │
        │ ✓ query: str                        │
        │ ✓ max_price: float                  │
        │ ✓ min_rating: float [0-5]           │
        │ ✓ platforms: List[str]              │
        └─────────────────────────────────────┘
                          │
                          ↓
        ┌─────────────────────────────────────┐
        │ DataFrame Filtering                 │
        │ df[price ≤ 5000 &                   │
        │    rating ≥ 4.0 &                   │
        │    platform ∈ ["Amazon", "Flipk"]]  │
        │ Result: 150 products                │
        └─────────────────────────────────────┘
                          │
                          ↓
        ┌─────────────────────────────────────┐
        │ TF-IDF Keyword Matching             │
        │ For each of 150 products:           │
        │ cosine_sim(query, product_name)     │
        │ match_score ∈ [0, 1]                │
        └─────────────────────────────────────┘
                          │
                          ↓
        ┌─────────────────────────────────────┐
        │ normalize_scores()                  │
        │ value_score = (rating × reviews)    │
        │               / (price + ε)         │
        │ Clamp all scores to [0, 1]          │
        └─────────────────────────────────────┘
                          │
                          ↓
        ┌─────────────────────────────────────┐
        │ best_first_search()                 │
        │ h(n) = 0.35×value +                 │
        │        0.30×sentiment +             │
        │        0.20×match +                 │
        │        0.15×rating                  │
        │ Top 10 by heuristic_score           │
        └─────────────────────────────────────┘
                          │
                          ↓
        ┌─────────────────────────────────────┐
        │ LLM Enhancement                     │
        │ POST to Pollinations.ai             │
        │ Generate 2-sentence explanation     │
        │ for best_pick product               │
        └─────────────────────────────────────┘
                          │
                          ↓
        ┌─────────────────────────────────────┐
        │ SearchResult (Pydantic)             │
        │ {                                   │
        │   products: [10 ranked products]    │
        │   best_pick: products[0]            │
        │   ai_summary: "We found..."         │
        │   total_found: 150                  │
        │ }                                   │
        └─────────────────────────────────────┘
                          │
                          ↓
              ┌───────────────────┐
              │ JSON → Frontend   │
              │ Render UI         │
              └───────────────────┘
```

---

## Algorithm Comparison

```
┌────────────────────────────────────────────────────────────────────────┐
│ ALGORITHM        │ UNIT   │ PURPOSE          │ TIME      │ USE CASE   │
├────────────────────────────────────────────────────────────────────────┤
│ BFS              │ II     │ Level-by-level   │ O(V+E)    │ Category   │
│                  │        │ exploration      │           │ traversal  │
├────────────────────────────────────────────────────────────────────────┤
│ Best First       │ III    │ Heuristic-based  │ O(n log n)│ Product    │
│ Search           │        │ ranking          │           │ ranking    │
│                  │        │                  │           │ (MAIN USE) │
├────────────────────────────────────────────────────────────────────────┤
│ Hill Climbing    │ III    │ Local            │ O(n²) →   │ Constraint │
│                  │        │ optimization     │ O(n)      │ refinement  │
└────────────────────────────────────────────────────────────────────────┘

HEURISTIC FUNCTION (Best First Search):
h(n) = 0.35 × value_score +      (Price-to-quality ratio)
       0.30 × sentiment_score +  (Customer satisfaction)
       0.20 × match_score +      (Query relevance)
       0.15 × rating_norm        (Star rating)
       ─────
       1.00 (weights sum to 1)
```

---

## Key Connections Between Files

```
REQUEST PATH:
─────────────

1. Frontend
   └─ App.jsx
      └─ fetch(/search) → JSON body

2. Backend Receives
   └─ main.py:@app.post("/search")
      ├─ models.py: SearchQuery validation
      ├─ config.py: DATA_PATH
      └─ Load: processed_data_combined.csv

3. Processing
   ├─ Pandas: Filter DataFrame
   ├─ nlp_engine.py: TF-IDF matching
   ├─ algorithms.py: Best First Search
   │  └─ needs: normalize_scores()
   ├─ llm_service.py: Pollinations.ai call
   └─ models.py: SearchResult validation

4. Response
   ├─ JSON: products[], best_pick, ai_summary
   └─ App.jsx: setProducts(), setBestPick(), setAiSummary()

5. Display
   └─ App.jsx renders:
      ├─ AI Summary card
      ├─ Best Pick hero card
      └─ Product grid


OFFLINE SETUP:
──────────────

1. Raw Data
   ├─ dataset/amazon.csv
   ├─ (Flipkart CSV if available)
   └─ (Myntra CSV if available)

2. train_all_datasets.py
   ├─ scraper.py: Extract data
   ├─ nlp_engine.py: TF-IDF & sentiment
   ├─ algorithms.py: normalize_scores()
   └─ Save to trained_models/

3. Output
   ├─ processed_data_combined.csv
   ├─ tfidf_vectorizer.pkl
   ├─ scaler.pkl
   └─ training_stats.json
```

---

## Scoring Hierarchy

```
RAW DATA (from CSV)
├─ product_name
├─ price
├─ rating
├─ review_count
├─ reviews (text)
└─ platform
    │
    ├─ [OFFLINE PREP]
    │  ├─ sentiment_score ← VADER analysis of reviews (-1 to +1)
    │  └─ value_score ← (rating × log(reviews)) / price (0 to 1)
    │
    └─ [RUNTIME]
       ├─ match_score ← TF-IDF similarity to query (0 to 1)
       └─ heuristic_score ← weighted combination for ranking (0 to 1)

FINAL HEURISTIC (for ranking):
    h = 0.35 × value_score (0-1)
      + 0.30 × sentiment_score (-1 to +1) → normalized (0-1)
      + 0.20 × match_score (0-1)
      + 0.15 × rating_norm (0-1)
      = final score (0-1)

Products sorted by heuristic_score (descending)
Top product = "Best Pick"
```

---

## Configuration Pyramid

```
                        ┌──────────────────┐
                        │  vite.config.js  │
                        │  Proxy routes:   │
                        │  /search → :8000 │
                        └────────┬─────────┘
                                 │
                ┌────────────────┴────────────────┐
                │                                 │
        ┌───────▼────────┐            ┌────────▼────────┐
        │  Frontend      │            │  Backend        │
        │ :localhost     │            │ :127.0.0.1      │
        │  :5173         │            │ :8000           │
        └────────────────┘            └────────┬────────┘
                                               │
                                   ┌───────────▼───────────┐
                                   │  config.py            │
                                   ├─ DATA_PATH           │
                                   ├─ TFIDF_PATH          │
                                   ├─ SCALER_PATH         │
                                   ├─ DEFAULT_LIMIT       │
                                   └─ Logging setup       │
```

---

## Error Handling Flow

```
Frontend Error:
└─ Try fetch()
   ├─ Success → Parse JSON → setState()
   └─ Failure → Catch error → setError() → Display message

Backend Error:
├─ Invalid JSON → FastAPI 400 Bad Request
├─ Invalid schema → Pydantic 422 Unprocessable Entity
├─ Missing file → Use fallback_products()
├─ LLM timeout → Return fallback text (retry at 30s)
└─ Database error → HTTP 500 Internal Server Error

Pydantic Validation:
├─ Type mismatch → Return helpful error message
├─ Range violation (e.g., rating > 5) → Reject
└─ Missing required field → Specify which field
```

---

## Performance Characteristics

```
SEARCH OPERATION TIMING (45 matching products):
┌──────────────────────────────┬────────┐
│ Step                         │ Time   │
├──────────────────────────────┼────────┤
│ 1. CSV load & filter         │ 50ms   │
│ 2. Match score computation   │ 100ms  │
│ 3. Normalize scores          │ 20ms   │
│ 4. Best First Search         │ 30ms   │
│ 5. LLM API call              │ 2000ms │ ← BOTTLENECK (80%)
│ 6. JSON serialization        │ 10ms   │
├──────────────────────────────┼────────┤
│ TOTAL                        │ 2210ms │
└──────────────────────────────┴────────┘

MEMORY USAGE:
┌──────────────────────────────┬────────┐
│ Component                    │ Size   │
├──────────────────────────────┼────────┤
│ processed_data_combined.csv  │ 5MB    │
│ Pandas DataFrame in RAM      │ 50MB   │
│ TF-IDF vectorizer            │ 10MB   │
│ Scaler                       │ 1MB    │
├──────────────────────────────┼────────┤
│ TOTAL                        │ 66MB   │
└──────────────────────────────┴────────┘

Scaling:
├─ Single user: ~66MB RAM
├─ 10 concurrent users: ~660MB RAM (1 server)
└─ 100+ users: Need load balancer + multiple instances
```

---

## Integration Testing Checklist

```
✓ Data Preparation
  ├─ train_all_datasets.py completes successfully
  ├─ processed_data_combined.csv created
  ├─ tfidf_vectorizer.pkl created
  └─ training_stats.json created

✓ Backend Startup
  ├─ config.py loads successfully
  ├─ CSV file loads into Pandas
  ├─ Vectorizer and scaler load
  └─ Uvicorn server starts on :8000

✓ Frontend Startup
  ├─ Vite dev server starts on :5173
  ├─ Proxy routes configured
  └─ React App mounts

✓ API Request/Response
  ├─ SearchQuery validation works
  ├─ /search endpoint accessible
  ├─ Products filtered correctly
  ├─ Best First Search ranks correctly
  ├─ LLM call succeeds or fails gracefully
  └─ SearchResult validation works

✓ UI Rendering
  ├─ Search results display
  ├─ Best Pick card shows
  ├─ AI summary appears
  ├─ Product grid renders
  └─ Sorting/filtering works

✓ Error Cases
  ├─ Invalid query → 422 error
  ├─ Missing CSV → Fallback products
  ├─ LLM timeout → Fallback text
  └─ Network error → Error message on UI
```

---

## Quick Deploy Guide

```bash
# 1. OFFLINE SETUP (one-time)
cd backend
python train_all_datasets.py
# Creates trained_models/ artifacts

# 2. BACKEND START
cd backend
python -m uvicorn main:app --reload --port 8000
# Check: http://127.0.0.1:8000/docs

# 3. FRONTEND START (new terminal)
cd frontend-app
npm install
npm run dev
# Check: http://localhost:5173

# 4. TEST SEARCH
# Open http://localhost:5173
# Type: "wireless headphones"
# Click Search
# Should see results in ~2 seconds
```

---

This quick reference provides a visual overview of the entire system architecture, data flow, and key components at a glance!

