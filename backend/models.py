"""
Pydantic data models for the AI Shopping Assistant.

Defines request/response schemas and product data structures for the
multi-stage pipeline: NLP parsing → BFS scraping → scoring → ranking → LLM intelligence.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class Product(BaseModel):
    """Product model with multi-stage scoring for recommendation ranking."""
    
    name: str
    price: float
    rating: float = Field(ge=0, le=5, description="Rating from 0-5")
    review_count: int
    platform: str = Field(description="Amazon / Flipkart / Myntra")
    url: str
    image_url: str
    reviews: List[str] = Field(default_factory=list, description="Max 5 review texts")
    
    # Multi-stage scoring
    sentiment_score: float = Field(default=0, ge=-1, le=1, description="Review sentiment: -1 to 1")
    match_score: float = Field(default=0, ge=0, le=1, description="TF-IDF match: 0 to 1")
    value_score: float = Field(default=0, ge=0, le=1, description="Rating/price ratio: 0 to 1")
    heuristic_score: float = Field(default=0, ge=0, le=1, description="Best-First ranking: 0 to 1")
    
    ai_explanation: str = Field(default="", description="LLM-generated reason for recommendation")
    
    class Config:
        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "name": "Sony WH-1000XM5",
                "price": 24990,
                "rating": 4.7,
                "review_count": 12400,
                "platform": "Amazon",
                "url": "https://amazon.in/...",
                "image_url": "https://images.../product.jpg",
                "reviews": ["Great sound quality", "Excellent ANC"],
                "sentiment_score": 0.85,
                "match_score": 0.92,
                "value_score": 0.88,
                "heuristic_score": 0.90,
                "ai_explanation": "Best choice for bass-heavy earphones under 1500"
            }
        }


class SearchQuery(BaseModel):
    """User search query with optional filters."""
    
    query: str = Field(description="Natural language search query")
    max_price: float = Field(default=100000, description="Maximum price filter")
    min_rating: float = Field(default=0, ge=0, le=5, description="Minimum rating filter")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum number of ranked products to return")
    platforms: List[str] = Field(
        default=["Amazon", "Flipkart", "Myntra"],
        description="E-commerce platforms to search"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "boAt earphones under 1500 bass heavy",
                "max_price": 1500,
                "min_rating": 4.0,
                "limit": 10,
                "platforms": ["Amazon", "Flipkart", "Myntra"]
            }
        }


class FilterRequest(BaseModel):
    """Request to filter products with price/rating constraints."""
    
    products: List[Product]
    max_price: float
    min_rating: float = Field(ge=0, le=5)
    
    class Config:
        arbitrary_types_allowed = True


class SearchResult(BaseModel):
    """Final ranked search results with AI recommendations."""
    
    products: List[Product] = Field(description="All matching products ranked by heuristic_score")
    best_pick: Optional[Product] = Field(default=None, description="Top-ranked recommendation")
    ai_summary: str = Field(description="LLM-generated summary of results and best pick reason")
    total_found: int = Field(description="Total products matched across platforms")
    
    class Config:
        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "products": [{"name": "Product 1", "price": 1000}],
                "best_pick": {"name": "Product 1", "price": 1000},
                "ai_summary": "Sony WH-1000XM5 is the best choice due to superior noise cancellation and sound quality...",
                "total_found": 24
            }
        }
