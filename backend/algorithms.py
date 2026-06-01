

import math
import numpy as np
from typing import List, Dict, Any
from collections import deque
from heapq import nlargest


# ============================================================================
# ALGORITHM 1: BFS (BREADTH-FIRST SEARCH) — Unit II: Uninformed Search
# ============================================================================

def bfs_category_search(graph: Dict[str, Any], target_category: str) -> List[Dict[str, Any]]:
   
    print("\n" + "="*70)
    print(f"BFS: Starting category search for '{target_category}'")
    print("="*70)
    
    queue = deque()
    visited = set()
    results = []
    level_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    # Level 0: Initialize with platform nodes
    platforms = ["Amazon", "Flipkart", "Myntra"]
    for platform in platforms:
        if platform in graph:
            queue.append(("platform", platform, graph[platform], 0))
            level_counts[0] += 1
    
    print(f"Level 0 (Platforms): Initialized with {level_counts[0]} platform nodes")
    
    # BFS traversal
    while queue:
        node_type, node_name, node_data, level = queue.popleft()
        
        # Skip if visited
        if node_name in visited:
            continue
        visited.add(node_name)
        
        # Level 1: Process categories
        if level == 0 and isinstance(node_data, dict):
            for category, subcats in node_data.items():
                if isinstance(subcats, dict):
                    queue.append(("category", category, subcats, 1))
                    level_counts[1] += 1
            print(f"  Platform '{node_name}': Found {len(node_data)} categories at Level 1")
        
        # Level 2: Process subcategories
        elif level == 1 and isinstance(node_data, dict):
            for subcat, products in node_data.items():
                if isinstance(products, list):
                    queue.append(("subcategory", subcat, products, 2))
                    level_counts[2] += 1
        
        # Level 3: Check products
        elif level == 2 and isinstance(node_data, list):
            for product in node_data:
                if isinstance(product, dict):
                    # Check if category matches
                    prod_category = product.get("category", "").lower()
                    if target_category.lower() in prod_category:
                        results.append(product)
                        level_counts[3] += 1
    
    # Print traversal summary
    print(f"\nBFS Traversal Summary:")
    print(f"  Level 0 (Platforms):      {level_counts[0]} visited")
    print(f"  Level 1 (Categories):     {level_counts[1]} visited")
    print(f"  Level 2 (Subcategories):  {level_counts[2]} visited")
    print(f"  Level 3 (Products):       {level_counts[3]} matched")
    print(f"\nTotal products found: {len(results)}")
    print("="*70)
    
    return results


# ============================================================================
# ALGORITHM 2: BEST FIRST SEARCH — Unit III: Informed Search (Heuristic)
# ============================================================================

def best_first_search(products: List[Dict[str, Any]], top_n: int = 10) -> List[Dict[str, Any]]:
   
    print("\n" + "="*70)
    print(f"BEST FIRST SEARCH: Ranking {len(products)} products")
    print("="*70)
    
    if not products:
        print("No products to rank.")
        return []
    
    # Compute heuristic for each product
    products_with_h = []
    
    for product in products:
        # Extract scores (with defaults)
        value_score = product.get("value_score", 0.5)
        sentiment_score = product.get("sentiment_score", 0.0)
        match_score = product.get("match_score", 0.5)
        rating = product.get("rating", 2.5)
        
        # Normalize rating to 0-1 range
        rating_norm = rating / 5.0
        
        # Compute heuristic h(n)
        h_score = (
            0.35 * value_score +
            0.30 * sentiment_score +
            0.20 * match_score +
            0.15 * rating_norm
        )
        
        # Clamp between 0 and 1
        h_score = max(0.0, min(1.0, h_score))
        
        # Store heuristic back in product dict
        product["heuristic_score"] = h_score
        products_with_h.append((h_score, product))
    
    # Get top_n using heapq.nlargest
    top_products = nlargest(min(top_n, len(products_with_h)), products_with_h, key=lambda x: x[0])
    result = [product for _, product in top_products]
    
    # Print ranking summary
    print(f"\nHeuristic Weights:")
    print(f"  Value Score:    35%")
    print(f"  Sentiment:      30%")
    print(f"  Match Score:    20%")
    print(f"  Rating (norm):  15%")
    print(f"\nTop {top_n} Products Ranked by Heuristic:")
    for i, product in enumerate(result, 1):
        h = product.get("heuristic_score", 0)
        name = product.get("product_name", "Unknown")[:40]
        rating = product.get("rating", 0)
        print(f"  {i:2d}. h={h:.3f} | {name:40s} | Rating: {rating:.1f}")
    print("="*70)
    
    return result


# ============================================================================
# ALGORITHM 3: HILL CLIMBING — Unit III: Informed Search (Local Optimization)
# ============================================================================

def hill_climbing_filter(
    products: List[Dict[str, Any]],
    max_price: float,
    min_rating: float = 0.0
) -> List[Dict[str, Any]]:
   
    print("\n" + "="*70)
    print(f"HILL CLIMBING: Refining {len(products)} products")
    print(f"Constraints: max_price=${max_price:.2f}, min_rating={min_rating:.1f}")
    print("="*70)
    
    if not products:
        print("No products to filter.")
        return []
    
    # Helper to access dict-like or object attributes
    def _get(prod, key, default=None):
        if isinstance(prod, dict):
            return prod.get(key, default)
        return getattr(prod, key, default)

    # Helper to set value on dict-like or object
    def _set(prod, key, value):
        if isinstance(prod, dict):
            prod[key] = value
        else:
            try:
                setattr(prod, key, value)
            except Exception:
                pass

    # Filter products by constraints
    valid_products = []
    for product in products:
        price = _get(product, "price", float('inf'))
        rating = _get(product, "rating", 0)

        # Check constraints
        if price <= max_price and rating >= min_rating:
            valid_products.append(product)
    
    print(f"Constraint filtering: {len(products)} -> {len(valid_products)} products")
    
    if not valid_products:
        print(f"No products satisfy constraints.")
        print("="*70)
        return []
    
    # Hill climbing: iterate and optimize
    current_idx = 0
    current = valid_products[current_idx]
    current_h = _get(current, "heuristic_score", 0)
    moves = 0
    
    print(f"\nHill Climbing Traversal:")
    print(f"  Starting at product: {str(_get(current, 'product_name', 'Unknown'))[:40]}")
    print(f"  Initial h-score: {current_h:.3f}")
    
    # Explore neighbors
    improved = True
    while improved:
        improved = False
        
        # Check neighbors
        for next_idx in range(len(valid_products)):
            if next_idx == current_idx:
                continue

            neighbor = valid_products[next_idx]
            neighbor_h = _get(neighbor, "heuristic_score", 0)
            
            # Move to neighbor if better
            if neighbor_h > current_h:
                current = neighbor
                current_h = neighbor_h
                current_idx = next_idx
                moves += 1
                improved = True
                print(f"  Move {moves}: Found better neighbor, h={current_h:.3f}")
                break  # Move to this neighbor (greedy)
    
    if moves == 0:
        print(f"  No moves: Starting point is local maximum")
    else:
        print(f"  Stopped after {moves} moves (local maximum reached)")
    
    # Re-sort by heuristic score (descending)
    result = sorted(valid_products, key=lambda p: _get(p, "heuristic_score", 0), reverse=True)
    
    print(f"\nFinal results: {len(result)} products")
    for i, product in enumerate(result[:5], 1):
        h = _get(product, "heuristic_score", 0)
        name = str(_get(product, "product_name", "Unknown"))[:40]
        price = _get(product, "price", 0)
        rating = _get(product, "rating", 0)
        print(f"  {i}. h={h:.3f} | {name:40s} | ${price:.2f} | rating={rating:.1f}")
    if len(result) > 5:
        print(f"  ... and {len(result) - 5} more")
    
    print("="*70)
    return result


# ============================================================================
# HELPER FUNCTION: NORMALIZE SCORES
# ============================================================================

def normalize_scores(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize all numeric scores to 0-1 range using min-max normalization.
    
    Computes:
    - value_score = (rating_norm) * log(review_count+1) / (price_norm + 0.01)
    - Clamps all scores to [0, 1]
    
    Args:
        products: List of product dictionaries
    
    Returns:
        Same list with normalized score fields:
        - value_score, match_score, sentiment_score (if present)
    """
    print("\n" + "="*70)
    print("NORMALIZING SCORES: Converting to 0-1 range")
    print("="*70)
    
    if not products:
        return products
    
    # Extract values for min-max normalization
    prices = []
    ratings = []
    review_counts = []
    
    for product in products:
        price = product.get("price", 0)
        rating = product.get("rating", 0)
        review_count = product.get("rating_count", 0)
        
        if price > 0:
            prices.append(price)
        if rating > 0:
            ratings.append(rating)
        if review_count > 0:
            review_counts.append(review_count)
    
    # Calculate ranges
    min_price = min(prices) if prices else 1
    max_price = max(prices) if prices else 1
    min_rating = min(ratings) if ratings else 0
    max_rating = max(ratings) if ratings else 5
    max_reviews = max(review_counts) if review_counts else 1
    
    price_range = max_price - min_price if max_price > min_price else 1
    rating_range = max_rating - min_rating if max_rating > min_rating else 1
    
    # Normalize each product
    for product in products:
        # Get raw values
        price = product.get("price", 0)
        rating = product.get("rating", 0)
        review_count = product.get("rating_count", 0)
        
        # Normalize to 0-1
        if price_range > 0:
            price_norm = (price - min_price) / price_range
        else:
            price_norm = 0.5
        
        if rating_range > 0:
            rating_norm = (rating - min_rating) / rating_range
        else:
            rating_norm = 0.5
        
        # Compute value score
        log_reviews = math.log(review_count + 1)
        value_score = (rating_norm * log_reviews) / (price_norm + 0.01)
        value_score = max(0.0, min(1.0, value_score))  # Clip to [0, 1]
        
        # Store normalized scores
        product["price_norm"] = max(0.0, min(1.0, price_norm))
        product["rating_norm"] = max(0.0, min(1.0, rating_norm))
        product["value_score"] = value_score
        
        # Clamp match_score and sentiment_score if present
        if "match_score" in product:
            product["match_score"] = max(0.0, min(1.0, product["match_score"]))
        if "sentiment_score" in product:
            product["sentiment_score"] = max(-1.0, min(1.0, product["sentiment_score"]))
    
    print(f"\nNormalization Results:")
    print(f"  Price range: ${min_price:.2f} - ${max_price:.2f}")
    print(f"  Rating range: {min_rating:.1f} - {max_rating:.1f}")
    print(f"  Review count max: {max_reviews}")
    print(f"  Products normalized: {len(products)}")
    print("="*70)
    
    return products
