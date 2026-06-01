"""
NLP Engine for the AI Shopping Assistant.

Implements TF-IDF vectorization and VADER sentiment analysis for product understanding.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
from typing import List, Tuple


class TFIDFEngine:
    """TF-IDF based text vectorization and similarity."""
    
    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.tfidf_matrix = None
        self.documents = None
    
    def fit(self, documents: List[str]):
        """Fit TF-IDF vectorizer to documents."""
        self.documents = documents
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
    
    def get_similarities(self, query_idx: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """Get top-k similar documents to query document."""
        if self.tfidf_matrix is None:
            return []
        
        cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        similarities = list(enumerate(cosine_sim[query_idx].flatten()))
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        
        return similarities[1:top_k+1]  # Exclude self (first result)


class SentimentAnalyzer:
    """VADER sentiment analysis for product reviews."""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
    
    def analyze(self, text: str) -> dict:
        """Analyze sentiment of text."""
        scores = self.sia.polarity_scores(text)
        return {
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': scores['compound'],
            'sentiment': self._classify_sentiment(scores['compound'])
        }
    
    @staticmethod
    def _classify_sentiment(compound_score: float) -> str:
        """Classify sentiment based on compound score."""
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'


class NLPPipeline:
    """Complete NLP pipeline for product understanding."""
    
    def __init__(self):
        self.tfidf = TFIDFEngine()
        self.sentiment = SentimentAnalyzer()
    
    def process_products(self, products: List[dict]) -> List[dict]:
        """Process products with NLP analysis."""
        processed = []
        
        # Extract text features
        features = [f"{p.get('product_name', '')} {p.get('about_product', '')}" 
                   for p in products]
        
        # Fit TF-IDF
        self.tfidf.fit(features)
        
        # Analyze each product
        for i, product in enumerate(products):
            sentiment_analysis = self.sentiment.analyze(
                product.get('about_product', '')
            )
            
            processed.append({
                **product,
                'sentiment': sentiment_analysis['sentiment'],
                'sentiment_score': sentiment_analysis['compound']
            })
        
        return processed


def compute_match_score(query: str, text: str) -> float:
    """Compute a simple normalized match score between query and text.

    Uses token overlap; returns value in [0,1].
    """
    try:
        q_tokens = set(str(query).lower().split())
        t_tokens = set(str(text).lower().split())
        if not q_tokens or not t_tokens:
            return 0.0
        inter = len(q_tokens & t_tokens)
        union = len(q_tokens | t_tokens)
        return float(inter) / union if union > 0 else 0.0
    except Exception:
        return 0.0


def compute_sentiment(reviews: list) -> float:
    """Compute a sentiment score [-1, 1] for a list of review strings.

    Tries to use VADER if available; otherwise falls back to a simple
    lexicon-based heuristic.
    """
    try:
        sia = SentimentIntensityAnalyzer()
        text = " ".join(reviews)
        return float(sia.polarity_scores(text)['compound'])
    except Exception:
        # Fallback naive lexicon
        pos_words = {'good','great','excellent','love','best','awesome','nice','fantastic','amazing'}
        neg_words = {'bad','terrible','poor','worst','awful','hate','disappointed','disappointing','broken'}
        text = " ".join(reviews).lower()
        toks = text.split()
        if not toks:
            return 0.0
        pos = sum(1 for t in toks if t in pos_words)
        neg = sum(1 for t in toks if t in neg_words)
        score = (pos - neg) / max(1, len(toks))
        # normalize roughly to [-1,1]
        return max(-1.0, min(1.0, score * 2))


def score_all_products(products: list, query: str = '') -> list:
    """Score all product dicts in-place with a `match_score` using compute_match_score."""
    for p in products:
        text = f"{p.get('product_name','')} {p.get('description', p.get('about_product',''))}"
        p['match_score'] = compute_match_score(query, text) if query else 0.0
    return products
