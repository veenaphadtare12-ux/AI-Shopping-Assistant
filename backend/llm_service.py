"""
LLM Service for the AI Shopping Assistant.

PURPOSE: 100% Free and Zero-Config LLM using Pollinations.ai
No API Keys required!
"""

import requests
from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Safe getter that works for dicts and objects."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    try:
        return getattr(obj, key, default)
    except Exception:
        pass
    return default


def query_free_ai(prompt: str) -> str:
    """Send prompt to Pollinations AI (Free, No Key Required)."""
    try:
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful, professional AI shopping assistant. Keep your answers concise, direct, and exactly match the length requested."
                },
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(
            "https://text.pollinations.ai/",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            text = response.text.strip()
            # Clean up in case model echoes prompt tokens
            if "[INST]" in text:
                text = text.split("[/INST]")[-1].strip()
            return text

        logger.error(f"Pollinations API error: {response.status_code} - {response.text}")
        return ""

    except requests.exceptions.Timeout:
        logger.warning("Pollinations API timed out")
        return ""

    except Exception as e:
        logger.error(f"Pollinations API error: {e}")
        return ""


def explain_recommendation(product: Dict[str, Any], query: str) -> str:
    """Generate AI explanation for why this product is recommended."""
    name = _get(product, "product_name") or _get(product, "name", "Product")
    price = _get(product, "discounted_price", _get(product, "price", 0))
    rating = _get(product, "rating", 0)
    review_count = _get(product, "review_count", _get(product, "rating_count", 0))
    platform = _get(product, "platform", "Unknown")

    prompt = (
        f"Explain why this product is a great recommendation for someone searching '{query}' in exactly 2 sentences.\n\n"
        f"Product: {name}\n"
        f"Price: Rs.{price}\n"
        f"Rating: {rating} stars ({review_count} reviews)\n"
        f"Platform: {platform}\n\n"
        f"Recommendation (2 sentences only):"
    )

    result = query_free_ai(prompt)

    if not result:
        # Informative fallback using real product data
        return (
            f"{name} is an excellent choice for {query}, rated {rating} stars "
            f"by {review_count} verified buyers on {platform}. "
            f"Priced at Rs.{price}, it offers strong value for the quality delivered."
        )

    logger.info(f"Generated explanation for {name}")
    return result


def summarize_reviews(reviews: List[str], product_name: str) -> str:
    """Summarize customer reviews into 2 short insights."""
    if not reviews:
        return "Customers have shared positive feedback about this product."

    reviews_text = " | ".join(reviews[:5])

    prompt = (
        f"Summarize these customer reviews for '{product_name}' in exactly 2 short lines. "
        f"No bullet points, no bold text, just plain sentences:\n\n"
        f"Reviews: {reviews_text}\n\n"
        f"Summary (2 lines):"
    )

    result = query_free_ai(prompt)

    if not result:
        return "Customers have shared positive feedback about this product."

    return result


def compare_products(product1: Dict[str, Any], product2: Dict[str, Any]) -> str:
    """Compare two products and recommend which to buy."""
    p1_name = _get(product1, "product_name", _get(product1, "name", "Product 1"))
    p1_price = _get(product1, "discounted_price", _get(product1, "price", 0))
    p1_rating = _get(product1, "rating", 0)
    p1_platform = _get(product1, "platform", "Unknown")

    p2_name = _get(product2, "product_name", _get(product2, "name", "Product 2"))
    p2_price = _get(product2, "discounted_price", _get(product2, "price", 0))
    p2_rating = _get(product2, "rating", 0)
    p2_platform = _get(product2, "platform", "Unknown")

    prompt = (
        f"Compare these two products and clearly recommend which one to buy in exactly 3 sentences:\n\n"
        f"Product 1: {p1_name} — Rs.{p1_price}, {p1_rating} stars on {p1_platform}\n"
        f"Product 2: {p2_name} — Rs.{p2_price}, {p2_rating} stars on {p2_platform}\n\n"
        f"Comparison (3 sentences):"
    )

    result = query_free_ai(prompt)

    if not result:
        return (
            f"Both {p1_name} and {p2_name} are strong options in their price range. "
            f"Based on ratings, {p1_name if p1_rating >= p2_rating else p2_name} leads with better customer satisfaction."
        )

    return result


def generate_search_summary(query: str, total_found: int, best: Dict[str, Any]) -> str:
    """Generate a one-sentence AI summary of the search results."""
    best_name = _get(best, "product_name", _get(best, "name", "a top product"))
    best_price = _get(best, "discounted_price", _get(best, "price", 0))
    best_rating = _get(best, "rating", 0)

    prompt = (
        f"A user searched for: \"{query}\"\n"
        f"We found {total_found} products. The best recommendation is: "
        f"{best_name} at Rs.{best_price} with a rating of {best_rating} stars.\n\n"
        f"Write exactly one compelling sentence summarizing this search result for the user:"
    )

    result = query_free_ai(prompt)

    if not result:
        return (
            f"Found {total_found} products matching '{query}' — "
            f"{best_name} at Rs.{best_price} is your best value pick."
        )

    logger.info("Generated search summary")
    return result
