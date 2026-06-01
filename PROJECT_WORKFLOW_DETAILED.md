# 🏗️ AI Shopping Assistant - Complete Workflow & Architecture Guide

## 📋 Table of Contents
1. [High-Level Architecture](#high-level-architecture)
2. [Complete Data Flow Pipeline](#complete-data-flow-pipeline)
3. [Detailed File-by-File Explanation](#detailed-file-by-file-explanation)
4. [How Components Connect](#how-components-connect)
5. [The Search Journey (Step-by-Step)](#the-search-journey-step-by-step)
6. [Key Technologies & Why They're Used](#key-technologies--why-theyre-used)

---

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React)                             │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  App.jsx - Main React Component                            │     │
│  │  ├─ Search Input & Filters (price, rating, platforms)     │     │
│  │  ├─ State Management (query, products, bestPick, etc)     │     │
│  │  ├─ Fetch API Calls to Backend /search endpoint           │     │
│  │  └─ Beautiful Glassmorphic UI Rendering                   │     │
│  └────────────────────────────────────────────────────────────┘     │
│                              ↕ (JSON)                                │
└─────────────────────────────────────────────────────────────────────┘
                               ↓ HTTP POST
┌─────────────────────────────────────────────────────────────────────┐
│                        BACKEND (FastAPI)                             │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  main.py - API Server & Orchestration                      │     │
│  │  ├─ Loads processed_data_combined.csv on startup          │     │
│  │  ├─ Receives /search requests                             │     │
│  │  ├─ Performs keyword matching & filtering                 │     │
│  │  ├─ Calls algorithms (Best First Search, Hill Climbing)   │     │
│  │  ├─ Calls LLM Service for explanations                    │     │
│  │  └─ Returns ranked results + AI summary                   │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  models.py - Data Validation (Pydantic)                   │     │
│  │  ├─ Product (13 fields: scores, metadata, AI explanation)│     │
│  │  ├─ SearchQuery (user search + filters)                  │     │
│  │  ├─ SearchResult (final ranked output)                   │     │
│  │  └─ FilterRequest (for Hill Climbing refinement)         │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  algorithms.py - Search Algorithms (Academic Curriculum)  │     │
│  │  ├─ BFS (Breadth-First Search) - Unit II: Uninformed Search
│  │  │  └─ Hierarchical category traversal across platforms  │     │
│  │  ├─ Best First Search - Unit III: Informed Search        │     │
│  │  │  └─ Heuristic-based ranking (0-1 score)              │     │
│  │  └─ Hill Climbing - Unit III: Local Optimization         │     │
│  │     └─ Filter refinement with constraints                │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  llm_service.py - AI Text Generation                       │     │
│  │  ├─ Calls Pollinations.ai (Free LLM)                      │     │
│  │  ├─ Generates 2-sentence product explanations            │     │
│  │  ├─ Summarizes customer reviews                          │     │
│  │  └─ Creates overall search summary                       │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  nlp_engine.py - Natural Language Processing              │     │
│  │  ├─ TF-IDF Vectorization (keyword matching)               │     │
│  │  ├─ VADER Sentiment Analysis (review mood)                │     │
│  │  └─ Used during data preparation phase                   │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  config.py - Configuration                                │     │
│  │  ├─ DATA_PATH (processed_data_combined.csv location)      │     │
│  │  ├─ TFIDF_PATH, SCALER_PATH (trained models)             │     │
│  │  ├─ DEFAULT_LIMIT (20 results)                           │     │
│  │  └─ Logging configuration                                │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  ml_recommender.py - Lightweight Recommender Class        │     │
│  │  ├─ compute_match_scores() - TF-IDF similarity           │     │
│  │  └─ get_recommendations() - Full pipeline                │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  DATA: trained_models/processed_data_combined.csv         │     │
│  │  ├─ Pre-processed product data from Amazon, Flipkart,    │     │
│  │  ├─ Myntra with calculated scores (sentiment, value)     │     │
│  │  └─ Loaded into memory (Pandas DataFrame) on startup     │     │
│  └────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
                               ↓ HTTP Response
┌─────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React)                             │
│  Displays ranked products with Best Pick, AI summary, and          │
│  interactive sorting/filtering                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Complete Data Flow Pipeline

### **OFFLINE PHASE (Before Server Starts)**

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DATA SCRAPING & COLLECTION                              │
│    ├─ scraper.py (or pre-scraped CSV files)                │
│    └─ Collects products from:                              │
│       ├─ Amazon.csv (dataset/amazon.csv)                   │
│       ├─ Flipkart (if available)                           │
│       └─ Myntra (if available)                             │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. RAW DATASET: products_raw.csv                            │
│    ├─ product_name, price, rating, review_count, url, etc │
│    └─ No scores yet (sentiment, value, match)              │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. NLP PROCESSING (train_all_datasets.py)                   │
│    ├─ Extract text features (product name, description)    │
│    ├─ Compute TF-IDF vectors for keyword matching          │
│    ├─ Fit TF-IDF vectorizer & save to .pkl                 │
│    ├─ Fit StandardScaler for feature normalization         │
│    └─ Apply VADER sentiment analysis to reviews            │
│       └─ Result: sentiment_score (-1 to +1)                │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. SCORE CALCULATION (normalize_scores in algorithms.py)   │
│    ├─ value_score = (rating_norm × log(reviews)) /         │
│    │                (price_norm + 0.01)                    │
│    │  Represents: Price-to-quality ratio (0-1)             │
│    │                                                        │
│    └─ All scores clamped to [0, 1] range                   │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. PROCESSED DATASET: processed_data_combined.csv           │
│    ├─ product_name, price, rating, review_count, ...      │
│    ├─ sentiment_score (VADER analysis)                     │
│    ├─ value_score (computed)                               │
│    ├─ platform (Amazon/Flipkart/Myntra)                    │
│    ├─ url, image_url, category, description, reviews      │
│    └─ stored in: backend/trained_models/                   │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. ARTIFACTS SAVED FOR RUNTIME                              │
│    ├─ tfidf_vectorizer.pkl (for keyword matching)           │
│    ├─ scaler.pkl (for feature normalization)                │
│    ├─ training_stats.json (performance metrics)             │
│    └─ processed_data_combined.csv (main dataset)            │
└─────────────────────────────────────────────────────────────┘
```

### **RUNTIME PHASE (When User Searches)**

```
USER INTERFACE
       ↓
   Search Input:
   ├─ query: "wireless headphones under 5000"
   ├─ max_price: 5000
   ├─ min_rating: 4.0
   ├─ platforms: ["Amazon", "Flipkart"]
   └─ limit: 10
       ↓
     HTTP POST → /search endpoint
       ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: CONSTRAINT FILTERING (main.py)                      │
│ ├─ Load processed_data_combined.csv into Pandas DataFrame  │
│ ├─ Filter 1: price <= 5000                                 │
│ ├─ Filter 2: rating >= 4.0                                 │
│ ├─ Filter 3: platform in ["Amazon", "Flipkart"]           │
│ └─ Result: ~50-200 matching products                       │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: KEYWORD MATCHING (main.py)                          │
│ ├─ For each filtered product, compute match_score          │
│ ├─ Using TF-IDF similarity:                                │
│ │   - Vectorize query: "wireless headphones under 5000"   │
│ │   - Vectorize product_name + description                │
│ │   - Compute cosine_similarity (0-1 range)                │
│ ├─ match_score = cosine_similarity or fallback token ratio│
│ └─ Products now have: [price, rating, sentiment_score,    │
│    value_score, match_score]                              │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: HEURISTIC RANKING (Best First Search)              │
│         algorithms.best_first_search()                      │
│                                                             │
│ For each product, compute heuristic_score:                 │
│   h(n) = 0.35×value_score +                                │
│           0.30×sentiment_score +                           │
│           0.20×match_score +                               │
│           0.15×rating_norm                                 │
│                                                             │
│ Weights:                                                    │
│ • 35%: How good value is (price vs quality)                │
│ • 30%: Customer sentiment (review positivity)              │
│ • 20%: How well query matches product                      │
│ • 15%: Overall rating (out of 5)                           │
│                                                             │
│ Result: Products sorted by heuristic_score (descending)    │
│ Top product becomes "Best Pick"                            │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: LLM ENHANCEMENT (llm_service.py)                    │
│ ├─ For Best Pick product:                                  │
│ │   Call query_free_ai() → Pollinations.ai                │
│ │   Prompt: "Why is this product best for user's query?"  │
│ │   Generate: 2-sentence AI explanation                    │
│ │   Result: ai_explanation field populated                │
│ │                                                          │
│ ├─ Generate overall search summary:                        │
│ │   Summarize top 3-5 products + query                    │
│ │   Create concise recommendation insight                 │
│ │   Result: ai_summary field                              │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: RESPONSE BUILDING (main.py)                         │
│ ├─ Format top 10 (or limit) products as JSON               │
│ ├─ Include all scores & metadata                           │
│ ├─ Set best_pick = products[0] (highest heuristic_score)  │
│ ├─ Include ai_summary                                      │
│ ├─ Include total_found (count before limit)                │
│ └─ Validate with Pydantic SearchResult model               │
└────────────────────────┬────────────────────────────────────┘
                         ↓
     HTTP Response (JSON)
       ↓
     FRONTEND RENDERS:
     ├─ Best Pick Card (hero product with AI explanation)
     ├─ AI Summary (search insight)
     └─ Product Grid (remaining ranked products)
```

---

## 📁 Detailed File-by-File Explanation

### **BACKEND FILES**

#### **1. `main.py` - The API Server & Orchestrator**

**Purpose**: Entry point of the FastAPI application. Handles all HTTP requests and orchestrates the search pipeline.

**Key Functions**:
```python
@app.get("/")                    # Health check
@app.post("/search")             # Main search endpoint
@app.post("/compare")            # Product comparison endpoint
@app.post("/filter")             # Apply Hill Climbing filter
```

**Search Endpoint Flow**:
1. **Receives** `SearchQuery` JSON with query, filters, platforms
2. **Validates** data using Pydantic model (enforces types, ranges)
3. **Loads** `processed_data_combined.csv` into Pandas DataFrame (on first startup, cached in memory)
4. **Applies Constraints**: 
   - Filter by price: `df[df['price'] <= max_price]`
   - Filter by rating: `df[df['rating'] >= min_rating]`
   - Filter by platform: `df[df['platform'].isin(platforms)]`
5. **Computes Match Scores**: Uses TF-IDF or fallback token overlap to score each product's relevance to query
6. **Calls** `best_first_search()` algorithm to rank by composite heuristic
7. **Calls** `llm_service.explain_recommendation()` to generate AI text for top product
8. **Returns** `SearchResult` object: products list, best_pick, ai_summary, total_found

**Key Code Pattern**:
```python
# Load data
df = pd.read_csv(DATA_PATH)

# Filter
df = df[(df['price'] <= max_price) & (df['rating'] >= min_rating)]

# Convert to dicts for algorithm
products = df.to_dict(orient='records')

# Score
products = normalize_scores(products)

# Rank
ranked = best_first_search(products, top_n=limit)

# Generate AI text
ranked[0]['ai_explanation'] = llm_service.explain_recommendation(ranked[0], query)

# Return
return SearchResult(products=ranked, best_pick=ranked[0], ai_summary=..., total_found=len(df))
```

**Why This Architecture?**
- **FastAPI** chosen for speed, async support, and automatic Pydantic integration
- **CORS middleware** allows frontend to make cross-origin requests
- **In-memory DataFrame** provides O(1) startup, O(n) search time
- **Separation of concerns**: algorithms, LLM, data models are imported as separate modules

---

#### **2. `algorithms.py` - Search Algorithms (Core AI Logic)**

**Purpose**: Implements three academic AI search algorithms for e-commerce product ranking.

**Algorithm 1: BFS (Breadth-First Search) - Unit II**

```
Function: bfs_category_search(graph, target_category)

Structure:
  Platforms (Level 0)
    └─ Categories (Level 1)
         └─ Subcategories (Level 2)
              └─ Products (Level 3)

Example:
  graph = {
    "Amazon": {
      "Electronics": {
        "Phones": [product1, product2, ...],
        "Headphones": [product3, ...]
      }
    }
  }

Algorithm:
  1. Initialize queue with ["Amazon", "Flipkart", "Myntra"]
  2. Process nodes level-by-level (not depth-first)
  3. For each platform, extract categories → subcategories → products
  4. Return all products in target category across all platforms

Complexity: O(P + E) where P = products, E = category edges
Best For: Hierarchical product discovery, exploring all options at same level
```

**Algorithm 2: Best First Search - Unit III (MOST IMPORTANT)**

```
Function: best_first_search(products, top_n=10)

Heuristic Function: h(n) = 0.35×value + 0.30×sentiment + 0.20×match + 0.15×rating

Score Breakdown:
  • value_score (35%):    price-to-quality ratio
                          High rating + low price = high value
                          formula: (rating_norm × log(reviews)) / (price_norm + 0.01)
  
  • sentiment_score (30%): from VADER sentiment analysis on reviews
                          Range: -1 (negative) to +1 (positive)
                          Indicates customer satisfaction
  
  • match_score (20%):    TF-IDF cosine similarity
                          How well user's query matches product name/description
                          Range: 0-1
  
  • rating_norm (15%):    normalized out-of-5 rating
                          Range: 0-1 (0 = 0 stars, 1 = 5 stars)

Algorithm:
  1. For each product, compute h(n) = weighted sum of scores
  2. Use heapq.nlargest() to efficiently get top N products
  3. Return products sorted by heuristic score (descending)

Example:
  Product A: h = 0.35(0.8) + 0.30(0.7) + 0.20(0.9) + 0.15(0.8) = 0.785
  Product B: h = 0.35(0.6) + 0.30(0.8) + 0.20(0.7) + 0.15(0.9) = 0.710
  → Product A ranked higher
  
Complexity: O(n log n) for sorting
Why Used: Considers multiple dimensions of "best" product, not just price or rating
```

**Algorithm 3: Hill Climbing - Unit III**

```
Function: hill_climbing_filter(products, max_price, min_rating)

Purpose: Local optimization to refine already-ranked products with additional constraints

Algorithm:
  1. Start with first product (current_best)
  2. Iterate through all other products (neighbors)
  3. If neighbor has higher heuristic_score AND satisfies constraints:
       → Move to neighbor (greedy step)
       → Increment move counter
  4. Stop when no neighbor is better (local maximum reached)
  5. Re-sort final results by heuristic_score

Constraints:
  • price <= max_price
  • rating >= min_rating

Example Flow:
  Start: Product A (h=0.70)
    ↓ check Product B (h=0.65, price=3000 > max) → skip
    ↓ check Product C (h=0.85, price=2000, rating=4.2) → move!
  Current: Product C (h=0.85)
    ↓ check Product A (h=0.70) → lower, skip
    ↓ check Product B (h=0.65) → lower, skip
    ↓ check Product D (h=0.88, price=2500, rating=4.5) → move!
  Current: Product D (h=0.88)
    ↓ no better neighbor → STOP
  
  Result: [Product D, Product C, Product A, Product B] (sorted by h)

Complexity: O(n²) worst case (all products compared), but typically O(n) for early stopping
Why Used: Quick refinement after ranking, finds local optima faster than global optimization
```

**Helper Function: `normalize_scores(products)`**

```
Converts all numeric scores to [0, 1] range using min-max normalization

For each product:
  1. Extract raw values: price, rating, review_count
  2. Find min/max across all products
  3. Normalize using formula: (value - min) / (max - min)
  4. Clamp to [0, 1] to handle edge cases
  
Formula for value_score:
  value_score = (rating_norm × log(review_count + 1)) / (price_norm + 0.01)
  • Numerator: quality metric (rating × review credibility)
  • Denominator: cost metric (price with small offset to avoid division by zero)
  • Result: higher quality + more reviews + lower price = higher value_score

Why Important: Ensures all scores are comparable and in consistent range
```

---

#### **3. `models.py` - Data Validation (Pydantic)**

**Purpose**: Define and validate data structures for all API requests/responses.

**4 Pydantic Models**:

```python
# 1. PRODUCT MODEL (13 fields)
class Product(BaseModel):
    # Metadata from dataset
    name: str                              # Product name
    price: float                           # Selling price (₹)
    rating: float = Field(ge=0, le=5)    # Star rating (0-5)
    review_count: int                      # Number of reviews
    platform: str                          # Amazon/Flipkart/Myntra
    url: str                               # Product link
    image_url: str                         # Product image
    reviews: List[str]                     # Top 5 review texts
    
    # Scoring fields (calculated)
    sentiment_score: float = Field(default=0, ge=-1, le=1)
    match_score: float = Field(default=0, ge=0, le=1)
    value_score: float = Field(default=0, ge=0, le=1)
    heuristic_score: float = Field(default=0, ge=0, le=1)
    
    # AI explanation
    ai_explanation: str = Field(default="")

# Field validation ensures:
# - rating: must be 0 ≤ value ≤ 5
# - sentiment_score: must be -1 ≤ value ≤ 1
# - match_score, value_score: must be 0 ≤ value ≤ 1

# 2. SEARCHQUERY MODEL (user's request)
class SearchQuery(BaseModel):
    query: str                             # "wireless headphones under 5000"
    max_price: float = Field(default=100000)
    min_rating: float = Field(default=0, ge=0, le=5)
    limit: int = Field(default=20, ge=1, le=100)
    platforms: List[str] = Field(default=["Amazon", "Flipkart", "Myntra"])

# 3. FILTERREQUEST MODEL (for Hill Climbing refinement)
class FilterRequest(BaseModel):
    products: List[Product]                # Pre-ranked products
    max_price: float                       # New price constraint
    min_rating: float                      # New rating constraint

# 4. SEARCHRESULT MODEL (final response)
class SearchResult(BaseModel):
    products: List[Product]                # All matching products
    best_pick: Optional[Product]           # Top recommendation
    ai_summary: str                        # Overall summary text
    total_found: int                       # Total matches before limit
```

**Why Pydantic?**
- **Type Safety**: Enforces correct data types (no string prices)
- **Range Validation**: Ensures ratings are 0-5, scores are 0-1
- **Auto Conversion**: Converts "4.5" (string) → 4.5 (float)
- **Error Responses**: Automatically generates helpful error messages
- **FastAPI Integration**: Automatically generates API documentation

**Example Flow**:
```
Frontend sends:
{
  "query": "headphones",
  "max_price": 5000,
  "min_rating": "not_a_number"
}

Pydantic validation fails →
FastAPI returns 422 error:
{
  "detail": [
    {
      "loc": ["body", "min_rating"],
      "msg": "value is not a valid float",
      "type": "type_error.float"
    }
  ]
}

Frontend sends valid data →
SearchQuery object created →
All fields validated →
main.py function receives validated object
```

---

#### **4. `llm_service.py` - AI Text Generation**

**Purpose**: Generate human-readable product explanations and summaries using free LLM.

**Key Functions**:

```python
# 1. query_free_ai(prompt: str) → str
#    Sends prompt to Pollinations.ai (free, no API key required)
#    Returns AI-generated response text
#    Timeout: 30 seconds

def query_free_ai(prompt):
    """
    POST to https://text.pollinations.ai/
    
    Payload:
    {
      "messages": [
        {"role": "system", "content": "You are a helpful shopping assistant..."},
        {"role": "user", "content": prompt}
      ]
    }
    
    Response: Plain text from LLM
    
    Error Handling:
    - Timeout after 30s → fallback response
    - API error → fallback response
    - Network error → fallback response
    """

# 2. explain_recommendation(product, query) → str
#    Generates 2-sentence recommendation explanation
#    
#    Prompt Template:
#    "Explain why this product is recommended for the user's search in 2 sentences.
#     Product: {name}
#     Price: ₹{price}
#     Rating: {rating}★ ({reviews} reviews)
#     Platform: {platform}
#     User Query: {query}
#     Recommendation (exactly 2 sentences):"
#    
#    Example Output:
#    "Sony WH-1000XM5 offers industry-leading noise cancellation perfect for 
#     your search with a stellar 4.7★ rating from 12,400 verified reviews. 
#     Its exceptional audio quality and 30-hour battery life make it an 
#     excellent value for the price."

# 3. summarize_reviews(reviews, product_name) → str
#    Summarizes top 5 customer reviews in 2 lines
#    
#    Prompt:
#    "Summarize these customer reviews for {product_name} in 2 short lines:
#     Reviews: {review1} | {review2} | ..."
#    
#    Example Output:
#    "Customers praise the exceptional noise cancellation and comfort.
#     Some mention the high price but confirm premium build quality."

# 4. search_summary(query, best_pick, products) → str
#    Creates overall search result summary
#    
#    Includes:
#    - What user searched for
#    - Why best_pick is recommended
#    - Overall market insights
```

**Why Pollinations.ai?**
- **Free**: No API key or credit card required
- **No Rate Limiting**: Unlike OpenAI/Anthropic free tiers
- **Good Quality**: Uses Meta's Llama model
- **Simple API**: Just POST JSON, get text back

**Error Handling**:
```
Success → Return AI-generated text
Timeout (>30s) → Return pre-written fallback text
Network error → Return fallback text
API error → Return fallback text

Fallback: "{name} is highly rated at {rating}★ with {reviews} reviews."
```

---

#### **5. `config.py` - Configuration**

**Purpose**: Centralized configuration for paths, constants, and logging.

```python
import logging
from pathlib import Path

ROOT = Path(__file__).parent  # /backend/

# Dataset path
DATA_PATH = ROOT / "trained_models" / "processed_data_combined.csv"
# Points to: /backend/trained_models/processed_data_combined.csv

# ML artifacts
TFIDF_PATH = ROOT / "trained_models" / "tfidf_vectorizer.pkl"
# Trained TF-IDF vectorizer for keyword matching

SCALER_PATH = ROOT / "trained_models" / "scaler.pkl"
# StandardScaler for feature normalization

# API defaults
DEFAULT_LIMIT = 20  # Return top 20 products

# Logging
LOG_LEVEL = "INFO"
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("shopmind")
```

**Why Centralize?**
- Easy to change paths for different environments
- All logging in one place
- Constants reused across modules
- Follows 12-factor app principles

---

#### **6. `nlp_engine.py` - Natural Language Processing**

**Purpose**: Keyword extraction (TF-IDF) and sentiment analysis (VADER).

**Classes**:

```python
# 1. TFIDFEngine
class TFIDFEngine:
    def __init__(self, max_features=5000):
        # Create TF-IDF vectorizer
        # max_features: keep top 5000 words (ignore rare words)
    
    def fit(self, documents):
        # Learn vocabulary from training data
        # Create sparse matrix representation
    
    def get_similarities(self, query_idx, top_k=5):
        # Compute cosine similarity between query and all documents
        # Return top-K most similar documents
        
        # Formula: similarity = (query · document) / (|query| × |document|)

# 2. SentimentAnalyzer
class SentimentAnalyzer:
    def __init__(self):
        # Initialize VADER (Valence Aware Dictionary and sEntiment Reasoner)
        # VADER specifically designed for social media/product review text
    
    def analyze(self, text):
        # Returns: {positive, negative, neutral, compound, sentiment}
        # compound: -1 (very negative) to +1 (very positive)
        # sentiment: "positive" / "negative" / "neutral"

# 3. NLPPipeline
class NLPPipeline:
    def process_products(self, products):
        # End-to-end processing
        # Extracts features from product_name + description
        # Computes TF-IDF vectors
        # Analyzes review sentiment
        # Returns enriched product list
```

**TF-IDF Explained**:
```
Term Frequency (TF):
  How often a word appears in a document
  TF(word, doc) = (count of word in doc) / (total words in doc)
  
Inverse Document Frequency (IDF):
  How unique a word is across all documents
  IDF(word) = log(total documents / documents containing word)
  
TF-IDF Score:
  TF-IDF = TF × IDF
  High score = word is frequent in this doc and rare overall
  Low score = word is common across all docs (stop words)
  
Example:
  Query: "wireless headphones"
  Product A name: "Sony WH-1000XM5 Wireless Headphones"
  Product B name: "JBL Bluetooth Speaker"
  
  TF-IDF of "wireless" in A: High (present + specific to audio)
  TF-IDF of "wireless" in B: High (present + specific to audio)
  TF-IDF of "headphones" in A: Very High (specific term)
  TF-IDF of "headphones" in B: Very Low (not present)
  
  Similarity(query, A) > Similarity(query, B)
```

**VADER Sentiment**:
```
Analyzes product reviews to gauge customer satisfaction

Example Reviews:
  "Amazing product! 10/10" → compound ≈ 0.95 (very positive)
  "Good value for money" → compound ≈ 0.68 (positive)
  "Average quality" → compound ≈ 0.0 (neutral)
  "Terrible, stopped working" → compound ≈ -0.75 (negative)
  "Worst purchase ever!!" → compound ≈ -0.95 (very negative)

Formula (simplified):
  sentiment_score = average(compound scores of all reviews)
  Range: -1 to +1

Why VADER for reviews?
  - Handles emojis, exclamation marks, capitalization
  - Designed for short texts (reviews are short)
  - No training needed (rule-based)
```

---

#### **7. `ml_recommender.py` - Lightweight Recommender Class**

**Purpose**: Simple wrapper class for getting recommendations in tests.

```python
class Recommender:
    def __init__(self, data_path):
        # Load processed_data_combined.csv
        # Try to load tfidf_vectorizer.pkl and scaler.pkl
        # Ensure required columns exist
    
    def compute_match_scores(self, query, df_slice):
        # For each product in df_slice, compute relevance to query
        # Uses TF-IDF vectorizer if available
        # Fallback: token overlap (Jaccard similarity)
        # Returns: List of floats in [0, 1]
        
        # Token overlap fallback:
        # score = |query_tokens ∩ product_tokens| / |query_tokens ∪ product_tokens|
    
    def get_recommendations(self, query, max_price, min_rating, top_n):
        # Full pipeline:
        # 1. Filter by price/rating
        # 2. Compute match_scores
        # 3. Ensure sentiment_score and value_score exist
        # 4. Call best_first_search()
        # 5. Return SearchResult
```

**Used By**: Tests, not the main API (main.py does this directly).

---

### **FRONTEND FILES**

#### **1. `App.jsx` - Main React Component**

**Purpose**: Main UI component handling search, state management, and rendering.

**State Management**:
```javascript
const [query, setQuery] = useState('')              // User search text
const [maxPrice, setMaxPrice] = useState(5000)      // Price filter
const [minRating, setMinRating] = useState(0)       // Rating filter
const [platforms, setPlatforms] = useState({        // Platform checkboxes
  Amazon: true,
  Flipkart: true,
  Myntra: true
})
const [loading, setLoading] = useState(false)       // Search in progress
const [products, setProducts] = useState([])        // Result products
const [bestPick, setBestPick] = useState(null)      // Top recommendation
const [aiSummary, setAiSummary] = useState('')      // AI summary text
const [error, setError] = useState('')              // Error message
const [sortMethod, setSortMethod] = useState('relevance')  // Sort by option
```

**Key Functions**:

```javascript
// 1. handleSearch(e, directQuery)
//    ├─ Gets query from input or directQuery param
//    ├─ Gets active platforms (checked boxes)
//    ├─ Constructs JSON body:
//    │  {
//    │    query: "wireless headphones",
//    │    max_price: 5000,
//    │    min_rating: 4.0,
//    │    platforms: ["Amazon", "Flipkart"],
//    │    limit: 10
//    │  }
//    ├─ POST to /search endpoint (via vite proxy)
//    ├─ Parses response JSON
//    ├─ Updates state: products, bestPick, aiSummary
//    └─ Scrolls to results

// 2. handlePlatformChange(platform)
//    ├─ Toggle platform checkbox
//    └─ Update platforms state

// 3. handleChipClick(text)
//    ├─ Suggested search chip clicked
//    ├─ Set query to chip text
//    └─ Auto-trigger search

// 4. getSortedProducts()
//    ├─ Sorts products by selected method:
//    │  ├─ "relevance": keep as-is (API ranking)
//    │  ├─ "price-asc": sort price ascending
//    │  ├─ "price-desc": sort price descending
//    │  └─ "rating": sort rating descending
//    └─ Return sorted array (doesn't modify state)
```

**Rendering Structure**:
```
┌─ Navigation Bar
│  ├─ Logo "ShopMind"
│  ├─ Nav links: Deals, Compare, Trending, Watchlist
│  └─ "AI Powered" badge
│
├─ Hero Section
│  ├─ Tagline: "Smart AI Recommendations"
│  ├─ Title: "Find the perfect product in seconds"
│  ├─ Subtitle: Description text
│  └─ Search Box
│     ├─ Search icon
│     ├─ Input field
│     └─ Submit button
│
├─ Filters Section
│  ├─ Max Price slider
│  ├─ Min Rating slider
│  └─ Platform checkboxes (Amazon, Flipkart, Myntra)
│
├─ Results Section (if search completed)
│  ├─ AI Summary Card
│  │  └─ Generated summary text
│  │
│  ├─ Best Pick Card (hero product)
│  │  ├─ Product image
│  │  ├─ Product name
│  │  ├─ Price + rating
│  │  ├─ AI Explanation (2 sentences)
│  │  └─ "View on {platform}" button
│  │
│  ├─ Sort Dropdown
│  │  ├─ Relevance (default)
│  │  ├─ Price Low to High
│  │  ├─ Price High to Low
│  │  └─ Rating
│  │
│  └─ Product Grid (10 products)
│     ├─ Product Card 1
│     │  ├─ Image
│     │  ├─ Name
│     │  ├─ Price
│     │  ├─ Rating + review count
│     │  └─ Link to product
│     ├─ Product Card 2
│     └─ ...
│
└─ Footer (if exists)
```

**API Integration**:
```javascript
// Vite proxy configuration (vite.config.js):
// "/search" → "http://127.0.0.1:8000/search"
// This solves CORS issues during development

// Frontend code:
const res = await fetch('/search', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body)
})
const data = await res.json()

// Response structure:
{
  products: [
    {
      product_id: "...",
      product_name: "Sony WH-1000XM5",
      price: 24990,
      discounted_price: 24990,
      rating: 4.7,
      rating_count: 2450,
      platform: "Amazon",
      url: "https://amazon.in/...",
      image_url: "https://images/...",
      reviews: ["Great sound", "Excellent ANC"],
      sentiment_score: 0.85,
      value_score: 0.78,
      heuristic_score: 0.87,
      ai_explanation: "Best choice for..."
    },
    ...
  ],
  best_pick: { ... },  // Same as products[0]
  ai_summary: "We found 47 products matching your search...",
  total_found: 47
}
```

---

#### **2. `index.css` - Glassmorphic Styling**

**Purpose**: Modern, beautiful UI with glassmorphism design.

**Key CSS Features**:

```css
/* Glassmorphism effect */
.card {
  background: rgba(255, 255, 255, 0.1);     /* Semi-transparent white */
  backdrop-filter: blur(10px);               /* Blur background */
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 20px;
}

/* CSS Variables for theming */
:root {
  --primary: #8B5CF6;        /* Vibrant purple */
  --secondary: #EC4899;      /* Pink accent */
  --dark: #1E1B4B;           /* Very dark purple */
  --light: #F8FAFC;          /* Almost white */
  --gradient: linear-gradient(135deg, #8B5CF6, #EC4899);
}

/* Responsive Grid */
.product-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 1.5rem;
}

/* Animations */
@keyframes glow {
  0%, 100% { box-shadow: 0 0 20px rgba(139, 92, 246, 0.3); }
  50% { box-shadow: 0 0 40px rgba(139, 92, 246, 0.6); }
}

.hero { animation: glow 3s ease-in-out infinite; }

@keyframes slideUp {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}

.product-card { animation: slideUp 0.5s ease-out forwards; }

/* Hover effects */
.product-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
  transition: all 0.3s ease;
}
```

**Design Principles**:
- **Glassmorphism**: Frosted glass effect (blur + transparency)
- **Micro-interactions**: Hover animations, smooth transitions
- **Color Psychology**: Purple = creativity/intelligence, Pink = fun/energy
- **Typography**: Modern fonts, good contrast, readable
- **Responsive**: Works on mobile, tablet, desktop

---

#### **3. `vite.config.js` - Build Configuration**

**Purpose**: Configure Vite build tool and dev server.

```javascript
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  
  server: {
    proxy: {
      '/search': 'http://127.0.0.1:8000',      // Frontend /search → Backend
      '/filter': 'http://127.0.0.1:8000',      // Frontend /filter → Backend
      '/compare': 'http://127.0.0.1:8000'      // Frontend /compare → Backend
    }
  }
})
```

**Why Proxy?**
- Frontend runs on `http://localhost:5173` (Vite dev server)
- Backend runs on `http://127.0.0.1:8000` (FastAPI)
- Different origins = CORS error
- Proxy routes requests through Vite, avoiding CORS

**Why Vite?**
- **Fast**: Uses ES modules natively, instant HMR (Hot Module Replacement)
- **Modern**: Native ES6 support, no complex webpack config
- **Development**: Starts server in <100ms vs Create React App's ~3s
- **Production**: Optimized build with code splitting and tree-shaking

---

#### **4. `index.html` - Entry Point**

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>ShopMind - AI Shopping Assistant</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

**Flow**:
1. Browser loads `index.html`
2. Runs `main.jsx` (entry point)
3. `main.jsx` renders `<App />` into `<div id="root">`
4. App component mounts and starts listening for user input

---

#### **5. `main.jsx` - React Entry Point**

```javascript
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
```

**Purpose**: Initialize React application and render root component.

---

### **DATA FILES**

#### **`trained_models/processed_data_combined.csv`**

**Purpose**: Pre-processed product dataset with calculated scores.

**Columns** (30+):
```
product_id          | Unique identifier
product_name        | "Sony WH-1000XM5 Wireless Headphones"
price               | 24990.0
discounted_price    | 24990.0 (if on sale)
currency            | "₹"
rating              | 4.7 (0-5 scale)
rating_count        | 12400 (number of reviews)
platform            | "Amazon" / "Flipkart" / "Myntra"
category            | "Electronics" / "Audio" / "Headphones"
subcategory         | "Wireless Headphones"
about_product       | Product description
url                 | Direct product link
image_url           | Product image URL
reviews             | Top 5 review texts (semicolon-separated)

# Pre-calculated scores:
sentiment_score     | -1 to +1 (VADER analysis of reviews)
value_score         | 0 to 1 (price/quality ratio)
match_score         | 0 to 1 (TF-IDF similarity to query) - calculated at runtime
heuristic_score     | 0 to 1 (final ranking score) - calculated at runtime
```

**Example Row**:
```
sony-wh-1000xm5 | Sony WH-1000XM5 Wireless Headphones | 24990.0 | 24990.0 | ₹ | 4.7 | 12400 | Amazon | Electronics | Audio | ... | "Great noise cancellation" | ... | 0.85 | 0.78 | ...
```

**Size**: ~2000-5000 products depending on dataset

---

### **TRAINING/PREPARATION SCRIPTS (OFFLINE)**

#### **`train_all_datasets.py` - Master Training Script**

**Purpose**: Orchestrates complete offline data preparation pipeline.

**Steps**:
1. **Load raw CSV files** from `dataset/amazon.csv`, etc.
2. **NLP Processing**:
   - Extract text features
   - Fit TF-IDF vectorizer (learn vocabulary)
   - Fit StandardScaler (feature normalization)
3. **Sentiment Analysis**:
   - Apply VADER to reviews
   - Calculate sentiment_score per product
4. **Score Calculation**:
   - Compute value_score using normalize_scores()
5. **Save Artifacts**:
   - `processed_data_combined.csv` → main dataset
   - `tfidf_vectorizer.pkl` → reused at runtime
   - `scaler.pkl` → reused at runtime
   - `training_stats.json` → performance metrics

**Output**: `backend/trained_models/` directory with all artifacts

---

#### **`train_models.py` & `scraper.py`**

- **`scraper.py`**: Scrapes product data from e-commerce sites (if needed)
- **`train_models.py`**: Individual model training (data transformation, feature engineering)

---

## 🔗 How Components Connect

### **Data Flow Diagram**

```
┌────────────────────────────────────────────────────────┐
│  OFFLINE PREPARATION (Before server starts)            │
│                                                        │
│  Raw CSV Files                                         │
│    ↓                                                   │
│  train_all_datasets.py                                 │
│    ├─ nlp_engine.py (TF-IDF, sentiment)               │
│    └─ algorithms.py (normalize_scores)                │
│    ↓                                                   │
│  Processed Data + Artifacts                            │
│  (trained_models/ directory)                          │
└────────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────┐
│  RUNTIME: Server Starts (Backend)                      │
│                                                        │
│  main.py:__init__()                                    │
│    ├─ Load config.py → set DATA_PATH                  │
│    ├─ Read processed_data_combined.csv                │
│    ├─ Load into Pandas DataFrame (in-memory)          │
│    ├─ Load tfidf_vectorizer.pkl for keyword matching  │
│    └─ Uvicorn server ready                            │
└────────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────┐
│  USER SEARCHES (Frontend sends HTTP POST)              │
│                                                        │
│  App.jsx:handleSearch()                                │
│    └─ fetch('/search', {...})                         │
│                         ↓                              │
│  vite.config.js (proxy)                                │
│    └─ Routes to http://127.0.0.1:8000/search         │
│                         ↓                              │
│  main.py:/search endpoint                              │
│    ├─ Validates request with models.py (Pydantic)     │
│    ├─ Filters DataFrame (price, rating, platform)    │
│    ├─ Computes match_scores using:                    │
│    │   ├─ tfidf_vectorizer (if available)             │
│    │   └─ Fallback: token overlap                     │
│    ├─ Calls algorithms.py:best_first_search()         │
│    │   └─ Computes heuristic_score for each product  │
│    ├─ Calls algorithms.py:normalize_scores()          │
│    │   └─ Ensures all scores in [0, 1]               │
│    ├─ Calls llm_service.py:explain_recommendation()   │
│    │   └─ Queries Pollinations.ai for AI text        │
│    ├─ Builds SearchResult with models.py             │
│    └─ Returns JSON response                           │
│                         ↓                              │
│  App.jsx: Receives response                            │
│    ├─ Parse JSON                                      │
│    ├─ Update state: products, bestPick, aiSummary    │
│    └─ Re-render UI                                    │
│                         ↓                              │
│  Browser: Display results                              │
│    ├─ Render Best Pick card                           │
│    ├─ Render AI Summary                               │
│    └─ Render product grid                             │
└────────────────────────────────────────────────────────┘
```

### **Module Dependencies**

```
main.py
  ├─ imports algorithms.py
  │  ├─ (uses best_first_search, hill_climbing_filter)
  │  ├─ depends on: numpy, heapq
  │  └─ pure Python, no external dataset access
  │
  ├─ imports models.py
  │  ├─ (uses SearchQuery, SearchResult, Product)
  │  ├─ depends on: pydantic
  │  └─ data validation only, no logic
  │
  ├─ imports llm_service.py
  │  ├─ (uses explain_recommendation, query_free_ai)
  │  ├─ depends on: requests (HTTP library)
  │  └─ calls external API (Pollinations.ai)
  │
  ├─ imports config.py
  │  ├─ (uses DATA_PATH, DEFAULT_LIMIT, logger)
  │  ├─ depends on: pathlib, logging
  │  └─ configuration constants
  │
  └─ depends on: pandas, fastapi, uvicorn
     (reads CSV, serves HTTP, validates requests)

ml_recommender.py (used by tests)
  ├─ imports algorithms.py, models.py
  ├─ imports ml_recommender from sklearn
  └─ provides get_recommendations() wrapper

nlp_engine.py (used during training, optional at runtime)
  ├─ imports sklearn (TF-IDF, cosine similarity)
  ├─ imports nltk (VADER sentiment)
  └─ provides TFIDFEngine, SentimentAnalyzer classes

Standalone (no dependencies):
  └─ config.py (just constants and logging)
```

---

## 🚀 The Search Journey (Step-by-Step Example)

**Scenario**: User searches for "boAt earphones under 2000 with bass"

### **Step 1: Frontend - User Input**
```
User types: "boAt earphones under 2000 with bass"
User sets: max_price=2000, min_rating=3.5, platforms=[Amazon]
User clicks: "Search" button
```

### **Step 2: Frontend - API Call**
```javascript
// App.jsx:handleSearch()
body = {
  query: "boAt earphones under 2000 with bass",
  max_price: 2000,
  min_rating: 3.5,
  platforms: ["Amazon"],
  limit: 10
}

fetch('/search', { method: 'POST', body: JSON.stringify(body) })
```

### **Step 3: Backend - Request Validation**
```python
# main.py receives request
request_data = await request.json()

# Pydantic validates
search_query = SearchQuery(**request_data)
# ✓ query is str
# ✓ max_price is float
# ✓ min_rating is float (3.5)
# ✓ platforms is list of strings
# ✓ limit is int (10)

# If any field is invalid → 422 error
```

### **Step 4: Backend - Load & Filter**
```python
# main.py:/search endpoint
df = pd.read_csv(config.DATA_PATH)
# Loads processed_data_combined.csv (2000 products from all platforms)

# Apply hard constraints
df = df[(df['price'] <= 2000) & 
        (df['rating'] >= 3.5) & 
        (df['platform'] == 'Amazon')]
# Result: 45 products match criteria
```

### **Step 5: Backend - Keyword Matching**
```python
# For each of 45 products, compute match_score
query = "boAt earphones under 2000 with bass"

for idx, product in df.iterrows():
    product_text = product['product_name']  # "boAt Rockerz 551"
    
    # Use TF-IDF vectorizer (trained offline)
    query_vector = tfidf_vectorizer.transform([query])
    product_vector = tfidf_vectorizer.transform([product_text])
    
    # Cosine similarity
    match_score = cosine_similarity(query_vector, product_vector)[0][0]
    # Result: 0.87 (high relevance)
    
    df.loc[idx, 'match_score'] = match_score

# After loop: all 45 products have match_score
```

### **Step 6: Backend - Best First Search**
```python
# algorithms.best_first_search(45 products)

for product in products:
    # h(n) = 0.35×value + 0.30×sentiment + 0.20×match + 0.15×rating
    
    value_score = product['value_score']           # e.g., 0.82
    sentiment_score = product['sentiment_score']   # e.g., 0.75
    match_score = product['match_score']           # e.g., 0.87
    rating = product['rating']                     # e.g., 4.2
    rating_norm = 4.2 / 5.0                        # 0.84
    
    h_score = (0.35 * 0.82 +
               0.30 * 0.75 +
               0.20 * 0.87 +
               0.15 * 0.84)
            = 0.287 + 0.225 + 0.174 + 0.126
            = 0.812  ← heuristic_score for this product
    
    product['heuristic_score'] = 0.812

# Get top 10 products (by heuristic_score)
# Result: 10 ranked products
```

### **Step 7: Backend - LLM Enhancement**
```python
# Best product is boAt Rockerz 551 (h=0.85)

best_product = ranked_products[0]

prompt = """Explain why this product is recommended for the user's search 
in exactly 2 sentences.

Product: boAt Rockerz 551
Price: ₹1299
Rating: 4.2★ (3621 reviews)
Platform: Amazon
User Query: boAt earphones under 2000 with bass

Recommendation (exactly 2 sentences):"""

# Call Pollinations.ai
ai_explanation = llm_service.query_free_ai(prompt)

# Response (simulated):
# "boAt Rockerz 551 delivers powerful bass response perfectly suited for your 
#  query, with an unbeatable price point of ₹1299. Its 4.2★ rating from over 
#  3,600 verified reviews confirms excellent value and reliable performance."

best_product['ai_explanation'] = ai_explanation
```

### **Step 8: Backend - Build Response**
```python
# models.SearchResult
response = SearchResult(
    products=[
        Product(
            name="boAt Rockerz 551",
            price=1299,
            rating=4.2,
            review_count=3621,
            platform="Amazon",
            url="https://amazon.in/...",
            image_url="https://...",
            reviews=["Great bass", "Value for money"],
            sentiment_score=0.75,
            match_score=0.87,
            value_score=0.82,
            heuristic_score=0.812,
            ai_explanation="boAt Rockerz 551 delivers..."
        ),
        # ... 9 more products
    ],
    best_pick=Product(...),  # Same as products[0]
    ai_summary="We found 45 boAt earphones matching your search. The boAt Rockerz 551 stands out with incredible bass...",
    total_found=45
)

# Convert to JSON and return
return response
```

### **Step 9: Frontend - Receive & Display**
```javascript
// App.jsx:handleSearch() - fetch completes
const data = await res.json()

// Parse response
setProducts(data.products)        // 10 products
setBestPick(data.best_pick)      // boAt Rockerz 551
setAiSummary(data.ai_summary)    // AI text

// Render:
// 1. AI Summary card: "We found 45 products..."
// 2. Best Pick hero card:
//    ├─ Product image
//    ├─ "boAt Rockerz 551"
//    ├─ "₹1299" + "4.2★ (3621 reviews)"
//    ├─ AI explanation: "boAt Rockerz 551 delivers..."
//    └─ "View on Amazon" button
// 3. Product grid with 9 more products
```

### **Step 10: User Sees Results**
```
┌─────────────────────────────────────────────────┐
│  ShopMind - AI Shopping Assistant               │
├─────────────────────────────────────────────────┤
│                                                 │
│  🔍 Search: boAt earphones under 2000...       │
│  📍 Filters: Max ₹2000 | Rating ≥3.5           │
│  🛍️ Platform: Amazon                            │
│                                                 │
├─ AI SUMMARY ──────────────────────────────────┤
│ "We found 45 products matching your search.   │
│  The boAt Rockerz 551 stands out with         │
│  incredible bass..."                           │
│                                                 │
├─ BEST PICK ───────────────────────────────────┤
│  [Image] boAt Rockerz 551                      │
│  ₹1,299  ⭐4.2 (3,621 reviews)                 │
│  AI: "boAt Rockerz 551 delivers powerful      │
│   bass response perfectly suited for your     │
│   query..."                                    │
│  [View on Amazon →]                           │
│                                                 │
├─ SIMILAR PRODUCTS ────────────────────────────┤
│  [Product Card] [Product Card] [Product Card]│
│  [Product Card] [Product Card] ...            │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 🛠️ Key Technologies & Why They're Used

### **Frontend Stack**

| Technology | Purpose | Why Chosen |
|------------|---------|-----------|
| **React** | UI framework | Component-based, reusable UI, state management |
| **Vite** | Build tool | Ultra-fast dev server, ES6 modules, instant HMR |
| **CSS3** | Styling | Custom glassmorphism, no dependency overhead |
| **JavaScript (ES6+)** | Logic | Async/await, fetch API, modern syntax |

### **Backend Stack**

| Technology | Purpose | Why Chosen |
|------------|---------|-----------|
| **FastAPI** | Web framework | Auto API docs, async support, Pydantic integration |
| **Uvicorn** | ASGI server | Lightweight, fast, production-ready |
| **Pandas** | Data manipulation | Fast CSV loading, filtering, vectorization |
| **Pydantic** | Data validation | Type hints, auto error messages, FastAPI native |
| **Scikit-learn** | ML tools | TF-IDF vectorization, cosine similarity |
| **NLTK** | NLP library | VADER sentiment analysis (pre-trained) |
| **Requests** | HTTP library | Simple LLM API calls |

### **AI/ML Tools**

| Tool | Purpose | Why Chosen |
|------|---------|-----------|
| **TF-IDF** | Keyword matching | Fast, interpretable, no training needed for inference |
| **VADER** | Sentiment analysis | Designed for short texts, no ML training required |
| **Best First Search** | Product ranking | Heuristic-based, considers multiple dimensions |
| **Hill Climbing** | Local optimization | Quick refinement with constraints |
| **Pollinations.ai** | LLM service | Free, no API key, good quality |

### **Data Format**

| Format | Usage |
|--------|-------|
| **CSV** | Persistent storage (processed_data_combined.csv) |
| **JSON** | API communication (requests/responses) |
| **PKL** | Scikit-learn artifacts (tfidf_vectorizer.pkl, scaler.pkl) |

---

## 📈 Performance Characteristics

### **Search Time Breakdown** (45 matching products)

```
1. Load CSV & filter:           ~50ms    (Pandas is fast)
2. Compute match_scores:         ~100ms   (TF-IDF similarity)
3. Normalize scores:             ~20ms    (Min-max normalization)
4. Best First Search:            ~30ms    (Sorting + heapq.nlargest)
5. LLM API call:                 ~2000ms  (Pollinations.ai network latency)
6. Response serialization:       ~10ms    (JSON encoding)
                                ─────────
Total:                          ~2200ms  (2.2 seconds user sees results)
```

**Bottleneck**: LLM API (80% of time). Solution: Cache LLM responses or generate async.

### **Memory Usage**

```
processed_data_combined.csv: ~5MB (2000 products × 30 columns)
Pandas DataFrame in RAM:     ~50MB (with indices, overhead)
TF-IDF vectorizer:           ~10MB (5000 vocabulary terms)
Scaler:                      ~1MB  (feature statistics)
                            ─────
Total:                      ~66MB

OK for single server, scale up with caching for high traffic.
```

---

## 🔐 Security & Stability

### **Input Validation**
- **Pydantic Models**: Ensure all inputs are correct type/range
- **SQL Injection**: Not applicable (no SQL queries)
- **CORS Policy**: Limited to trusted origins (currently "*")

### **Error Handling**
- **Missing Files**: Fallback products used if CSV not found
- **API Timeout**: 30-second timeout on LLM calls, returns fallback text
- **Invalid JSON**: FastAPI returns 422 error with details

### **Caching Opportunities**
- Cache TF-IDF vectorizer (already done)
- Cache processed dataset (already done in-memory)
- Cache LLM responses (not implemented, could reduce latency 80%)
- Cache product metadata (not needed, dataset is small)

---

## 🎓 Academic Integration

### **Curriculum Coverage**

| Algorithm | Unit | Concept | Implementation |
|-----------|------|---------|-----------------|
| **BFS** | Unit II | Uninformed Search | `algorithms.bfs_category_search()` |
| **Best First** | Unit III | Informed Search (Heuristic) | `algorithms.best_first_search()` |
| **Hill Climbing** | Unit III | Local Optimization | `algorithms.hill_climbing_filter()` |

### **Real-World Application**
- Not just theoretical algorithms, but practical e-commerce use case
- Demonstrates: algorithm selection, heuristic design, multi-criteria optimization
- Shows performance tradeoffs: accuracy vs speed vs complexity

---

## 🚦 How to Run the Full System

### **1. Prepare Data (Offline)**
```bash
cd backend
python train_all_datasets.py
# Creates: trained_models/processed_data_combined.csv, tfidf_vectorizer.pkl, scaler.pkl
```

### **2. Start Backend**
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
# Server starts on http://127.0.0.1:8000
# Loads processed dataset into memory (~1s startup)
```

### **3. Start Frontend**
```bash
cd frontend-app
npm install
npm run dev
# Vite dev server on http://localhost:5173
# Vite proxy routes /search → backend
```

### **4. Use Application**
```
1. Open http://localhost:5173 in browser
2. Type search query: "wireless headphones"
3. Adjust filters (price, rating, platforms)
4. Click "Search"
5. View Best Pick + AI Summary + Product Grid
```

---

This comprehensive guide covers the entire AI Shopping Assistant architecture, from data preparation through user interface rendering. Every file, algorithm, and connection is explained!

