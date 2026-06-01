"""
FastAPI backend for the AI Shopping Assistant.

Current behavior:
- Loads the processed multi-platform dataset
- Searches products with lightweight keyword matching
- Ranks results with the project algorithms
- Compares products using dataset-backed metadata plus LLM/rule-based text
"""

from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import llm_service
from algorithms import (
    best_first_search as algo_best_first_search,
    hill_climbing_filter as algo_hill_climbing_filter,
    normalize_scores as algo_normalize_scores,
)
from models import FilterRequest, SearchQuery, SearchResult
from config import DATA_PATH, DEFAULT_LIMIT, logger

app = FastAPI(title="AI Shopping Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _fallback_products() -> List[Dict[str, Any]]:
    """Fallback products used only if the processed dataset cannot be loaded."""
    return [
        {
            "product_id": "fallback_1",
            "name": "Sony WH-1000XM5 Wireless Headphones",
            "product_name": "Sony WH-1000XM5 Wireless Headphones",
            "price": 24990.0,
            "discounted_price": 24990.0,
            "rating": 4.7,
            "review_count": 2450,
            "rating_count": 2450,
            "platform": "Amazon",
            "url": "https://amazon.in/sony-headphones",
            "image_url": "https://images-na.ssl-images-amazon.com/sony-wh1000xm5.jpg",
            "reviews": ["Great sound quality", "Excellent ANC", "Comfortable for long use"],
            "category": "Audio",
            "description": "Wireless noise-cancelling headphones",
            "sentiment_score": 0.85,
            "value_score": 0.78,
        },
        {
            "product_id": "fallback_2",
            "name": "JBL Tune 750 Wireless Headphones",
            "product_name": "JBL Tune 750 Wireless Headphones",
            "price": 12999.0,
            "discounted_price": 12999.0,
            "rating": 4.2,
            "review_count": 1823,
            "rating_count": 1823,
            "platform": "Flipkart",
            "url": "https://flipkart.com/jbl-headphones",
            "image_url": "https://images-na.ssl-images-amazon.com/jbl-tune750.jpg",
            "reviews": ["Good bass", "Battery life is decent"],
            "category": "Audio",
            "description": "Wireless over-ear headphones",
            "sentiment_score": 0.72,
            "value_score": 0.74,
        },
        {
            "product_id": "fallback_3",
            "name": "boAt Rockerz 551 Wireless Headphones",
            "product_name": "boAt Rockerz 551 Wireless Headphones",
            "price": 2999.0,
            "discounted_price": 2999.0,
            "rating": 3.8,
            "review_count": 3621,
            "rating_count": 3621,
            "platform": "Amazon",
            "url": "https://amazon.in/boat-headphones",
            "image_url": "https://images-na.ssl-images-amazon.com/boat-rockerz551.jpg",
            "reviews": ["Budget friendly", "Average sound quality"],
            "category": "Audio",
            "description": "Budget wireless headphones",
            "sentiment_score": 0.65,
            "value_score": 0.81,
        },
    ]


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float parsing."""
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        text = str(value).replace(",", "").strip()
        try:
            return float(text)
        except ValueError:
            return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Best-effort integer parsing."""
    if pd.isna(value):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        text = str(value).replace(",", "").strip()
        try:
            return int(float(text))
        except ValueError:
            return default


def _load_products_from_dataset() -> List[Dict[str, Any]]:
    """Load products from the processed dataset into API-friendly dictionaries."""
    if not DATA_PATH.exists():
        logger.warning(f"Dataset not found at {DATA_PATH}. Using fallback products.")
        return _fallback_products()

    try:
        df = pd.read_csv(DATA_PATH).fillna("")
    except Exception as exc:
        logger.error(f"Failed to load dataset ({exc}). Using fallback products.")
        return _fallback_products()

    products = []
    for row in df.to_dict("records"):
        price = _safe_float(row.get("price", 0.0))
        rating = _safe_float(row.get("rating", 0.0))
        review_count = _safe_int(row.get("rating_count", 0))
        product_name = str(row.get("product_name", "")).strip()

        if not product_name:
            continue

        products.append(
            {
                "product_id": str(row.get("product_id", "")),
                "name": product_name,
                "product_name": product_name,
                "price": price,
                "discounted_price": price,
                "rating": rating,
                "review_count": review_count,
                "rating_count": review_count,
                "platform": str(row.get("platform", "")).strip(),
                "url": str(row.get("url", "")).strip(),
                "image_url": str(row.get("image_url", "")).strip(),
                "reviews": [],
                "category": str(row.get("category", "")).strip(),
                "description": str(row.get("description", "")).strip(),
                "seller": str(row.get("seller", "")).strip(),
                "sentiment_score": _safe_float(row.get("sentiment_score", 0.0)),
                "value_score": _safe_float(row.get("value_score", 0.0)),
            }
        )

    logger.info(f"Loaded {len(products)} products from dataset.")
    return products


ALL_PRODUCTS = _load_products_from_dataset()


def _tokenize(text: str) -> List[str]:
    """Split a user query into simple lowercase tokens."""
    return [token for token in text.lower().split() if token]


def _find_dataset_product(name: str, platform: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Find the best matching dataset product for a given name/platform."""
    if not name:
        return None

    target_name = name.strip().lower()
    target_platform = platform.strip().lower() if platform else None
    target_tokens = set(_tokenize(target_name))
    candidates = []

    for product in ALL_PRODUCTS:
        product_name = str(product.get("product_name", "")).lower()
        product_platform = str(product.get("platform", "")).lower()

        if target_platform and product_platform != target_platform:
            continue

        if product_name == target_name:
            score = 100
        elif target_name in product_name:
            score = 75
        else:
            overlap = len(target_tokens & set(_tokenize(product_name)))
            if not target_tokens:
                continue
            overlap_ratio = overlap / len(target_tokens)
            if overlap < 2 or overlap_ratio < 0.5:
                continue
            score = 40 + int(overlap_ratio * 20)

        candidates.append(
            (
                score,
                _safe_float(product.get("rating", 0.0)),
                _safe_int(product.get("review_count", product.get("rating_count", 0))),
                product,
            )
        )

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return candidates[0][3].copy()


def _build_compare_product(
    name: str,
    price: Optional[float],
    rating: Optional[float],
    platform: Optional[str],
) -> Dict[str, Any]:
    """Resolve comparison product from dataset first, then fall back to provided fields."""
    dataset_product = _find_dataset_product(name, platform)
    if dataset_product:
        if price is not None:
            dataset_product["price"] = price
            dataset_product["discounted_price"] = price
        if rating is not None:
            dataset_product["rating"] = rating
        return dataset_product

    resolved_price = price if price is not None else 0.0
    resolved_rating = rating if rating is not None else 0.0
    return {
        "product_name": name,
        "name": name,
        "price": resolved_price,
        "discounted_price": resolved_price,
        "rating": resolved_rating,
        "platform": platform or "Unknown",
        "value_score": 0.0,
        "sentiment_score": 0.0,
    }


def search_all_platforms(
    query: str,
    max_price: float,
    platforms: List[str],
    min_rating: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Search the loaded dataset with keyword matching across name/category/description.
    """
    query = query.strip().lower()
    tokens = _tokenize(query)
    selected_platforms = {platform.lower() for platform in platforms if platform}
    results = []

    for product in ALL_PRODUCTS:
        platform = str(product.get("platform", "")).lower()
        price = _safe_float(product.get("price", 0.0))
        rating = _safe_float(product.get("rating", 0.0))

        if selected_platforms and platform not in selected_platforms:
            continue
        if price > max_price:
            continue
        if rating < min_rating:
            continue

        searchable_text = " ".join(
            [
                str(product.get("product_name", "")),
                str(product.get("category", "")),
                str(product.get("description", "")),
                str(product.get("seller", "")),
            ]
        ).lower()

        if not tokens:
            match_ratio = 1.0
        else:
            matched_tokens = sum(1 for token in tokens if token in searchable_text)
            if matched_tokens == 0 and query not in searchable_text:
                continue
            match_ratio = matched_tokens / len(tokens) if matched_tokens else 0.5

        product_copy = product.copy()
        product_copy["match_score"] = max(0.0, min(1.0, match_ratio))
        results.append(product_copy)

    return results


def hill_climbing_filter(products: List[Dict[str, Any]], max_price: float, min_rating: float) -> List[Dict[str, Any]]:
    """Apply the real hill-climbing filter implementation from algorithms.py."""
    return algo_hill_climbing_filter(products, max_price=max_price, min_rating=min_rating)


@app.post("/search", response_model=SearchResult)
def search(query: SearchQuery):
    """Search products, rank them, and attach explanation/summary text."""
    logger.info(f"Received search query: {query.query}")
    try:
        raw_products = search_all_platforms(
            query=query.query,
            max_price=query.max_price,
            platforms=query.platforms,
            min_rating=query.min_rating,
        )

        if not raw_products:
            logger.info("No products found for query.")
            return SearchResult(
                products=[],
                best_pick=None,
                ai_summary=f"No products found for '{query.query}'.",
                total_found=0,
            )

        normalized = algo_normalize_scores(raw_products)
        ranked = algo_best_first_search(normalized, top_n=query.limit or DEFAULT_LIMIT)

        best = None
        if ranked:
            best = ranked[0].copy()
            product_name = best.get("product_name") or best.get("name", "Product")
            best["ai_explanation"] = llm_service.explain_recommendation(best, query.query)
            best["reviews_summary"] = llm_service.summarize_reviews(best.get("reviews", []), product_name)

        ai_summary = llm_service.generate_search_summary(
            query=query.query,
            total_found=len(ranked),
            best=best if best else {},
        )

        return SearchResult(
            products=ranked,
            best_pick=best,
            ai_summary=ai_summary,
            total_found=len(ranked),
        )

    except Exception as exc:
        logger.error(f"Error in /search route: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")


@app.post("/filter")
def filter_products(request: FilterRequest):
    """Filter products by price/rating constraints."""
    logger.info("Received filter request.")
    try:
        filtered = hill_climbing_filter(
            products=[product.model_dump() for product in request.products],
            max_price=request.max_price,
            min_rating=request.min_rating,
        )
        return {"products": filtered, "filtered_count": len(filtered)}

    except Exception as exc:
        logger.error(f"Error in /filter route: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Filter failed: {exc}")


@app.get("/compare")
def compare(
    p1_name: str,
    p2_name: str,
    p1_price: Optional[float] = None,
    p1_rating: Optional[float] = None,
    p1_platform: Optional[str] = None,
    p2_price: Optional[float] = None,
    p2_rating: Optional[float] = None,
    p2_platform: Optional[str] = None,
):
    """
    Compare two products using dataset-backed product lookup.

    If a name/platform pair is found in the processed dataset, the dataset row
    is used. Otherwise the provided query parameters are used directly.
    """
    logger.info(f"Received compare request: {p1_name} vs {p2_name}")
    try:
        p1 = _build_compare_product(p1_name, p1_price, p1_rating, p1_platform)
        p2 = _build_compare_product(p2_name, p2_price, p2_rating, p2_platform)
        comparison_text = llm_service.compare_products(p1, p2)

        return {
            "comparison": comparison_text,
            "product_1": {
                "name": p1.get("product_name") or p1.get("name"),
                "platform": p1.get("platform"),
                "price": p1.get("discounted_price", p1.get("price")),
                "rating": p1.get("rating"),
            },
            "product_2": {
                "name": p2.get("product_name") or p2.get("name"),
                "platform": p2.get("platform"),
                "price": p2.get("discounted_price", p2.get("price")),
                "rating": p2.get("rating"),
            },
        }

    except Exception as exc:
        logger.error(f"Error in /compare route: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Comparison failed: {exc}")


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "message": "Shopping assistant is running"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
