"""Lightweight ML recommender used by the test harness.

Provides a minimal `Recommender` class with methods:
- `compute_match_scores(query, df_slice)` -> list of floats
- `get_recommendations(query, max_price, min_rating, top_n)` -> SearchResult

The implementation uses available artifacts (`tfidf_vectorizer.pkl`, `scaler.pkl`)
if present, otherwise falls back to simple heuristics so tests can run.
"""
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import joblib

from .algorithms import best_first_search, normalize_scores
from .models import Product, SearchResult


class Recommender:
    def __init__(self, data_path: str = 'products_processed.csv'):
        self.root = Path(__file__).parent
        self.data_path = Path(data_path)

        if not self.data_path.exists():
            # try backend trained_models location
            alt = self.root / 'trained_models' / 'processed_data_combined.csv'
            if alt.exists():
                self.data_path = alt

        self.df = pd.read_csv(self.data_path)
        # Normalize column names: some datasets use 'rating_count' vs 'review_count'
        if 'review_count' not in self.df.columns and 'rating_count' in self.df.columns:
            self.df['review_count'] = self.df['rating_count']

        # load optional TF-IDF vectorizer
        self.vectorizer = None
        tfidf_path = Path('tfidf_vectorizer.pkl')
        if not tfidf_path.exists():
            alt = self.root / 'trained_models' / 'tfidf_vectorizer.pkl'
            if alt.exists():
                tfidf_path = alt

        if tfidf_path.exists():
            try:
                self.vectorizer = joblib.load(tfidf_path)
                print(f"Loaded TF-IDF vectorizer from {tfidf_path}")
            except Exception:
                self.vectorizer = None

        # load optional scaler
        self.scaler = None
        scaler_path = Path('scaler.pkl')
        if not scaler_path.exists():
            alt = self.root / 'trained_models' / 'scaler.pkl'
            if alt.exists():
                scaler_path = alt

        if scaler_path.exists():
            try:
                self.scaler = joblib.load(scaler_path)
                print(f"Loaded scaler from {scaler_path}")
            except Exception:
                self.scaler = None

        # Ensure value_score present
        if 'value_score' not in self.df.columns:
            products = self.df.to_dict(orient='records')
            normalize_scores(products)
            self.df = pd.DataFrame(products)

    def compute_match_scores(self, query: str, df_slice: pd.DataFrame) -> List[float]:
        """Return a list/array of match scores in [0,1] for rows in df_slice."""
        texts = []
        if 'product_name' in df_slice.columns and df_slice['product_name'].notna().any():
            texts = df_slice['product_name'].fillna('').astype(str).tolist()
        elif 'description' in df_slice.columns:
            texts = df_slice['description'].fillna('').astype(str).tolist()
        else:
            texts = [''] * len(df_slice)

        # If vectorizer is available, use cosine similarity
        if self.vectorizer is not None:
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                v_docs = self.vectorizer.transform(texts)
                v_query = self.vectorizer.transform([query])
                sims = cosine_similarity(v_query, v_docs).flatten()
                # clip to [0,1]
                sims = np.clip(sims, 0.0, 1.0)
                return sims.tolist()
            except Exception:
                pass

        # Fallback: simple token overlap normalized
        q_tokens = set(str(query).lower().split())
        scores = []
        for t in texts:
            toks = set(t.lower().split())
            if not toks or not q_tokens:
                scores.append(0.0)
                continue
            inter = len(q_tokens & toks)
            union = len(q_tokens | toks)
            scores.append(inter / union if union > 0 else 0.0)

        return scores

    def get_recommendations(self, query: str, max_price: float = 1e9, min_rating: float = 0.0, top_n: int = 5) -> SearchResult:
        """Run a simple pipeline to return top-N SearchResult for the given query."""
        df = self.df.copy()

        # Apply filters
        if 'price' in df.columns:
            df = df[df['price'] <= max_price]
        if 'rating' in df.columns:
            df = df[df['rating'] >= min_rating]

        if df.empty:
            return SearchResult(products=[], best_pick=None, ai_summary="", total_found=0)

        # Compute match scores
        match_scores = self.compute_match_scores(query, df)
        df = df.reset_index(drop=True)
        df['match_score'] = match_scores[:len(df)]

        # Ensure sentiment_score exists
        if 'sentiment_score' not in df.columns:
            df['sentiment_score'] = 0.0

        # Ensure value_score exists
        if 'value_score' not in df.columns:
            products = df.to_dict(orient='records')
            normalize_scores(products)
            df = pd.DataFrame(products)

        # Build product dicts expected by algorithms.best_first_search
        products = []
        for _, row in df.iterrows():
            prod = {
                'product_name': row.get('product_name') if 'product_name' in row else row.get('product_name', ''),
                'price': float(row.get('price', 0) or 0),
                'rating': float(row.get('rating', 0) or 0),
                'rating_count': int(row.get('rating_count', row.get('review_count', 0)) or 0),
                'platform': row.get('platform', ''),
                'url': row.get('url', ''),
                'image_url': row.get('image_url', ''),
                'sentiment_score': float(row.get('sentiment_score', 0) or 0),
                'match_score': float(row.get('match_score', 0) or 0),
                'value_score': float(row.get('value_score', 0) or 0),
            }
            products.append(prod)

        # Rank using best-first heuristic
        top_products = best_first_search(products, top_n=top_n)

        # Convert to Pydantic Product models
        product_models = []
        for p in top_products:
            prod_model = Product(
                name=p.get('product_name', '')[:250],
                price=p.get('price', 0.0),
                rating=p.get('rating', 0.0),
                review_count=p.get('rating_count', 0),
                platform=p.get('platform', ''),
                url=p.get('url', ''),
                image_url=p.get('image_url', ''),
                reviews=[],
                sentiment_score=p.get('sentiment_score', 0.0),
                match_score=p.get('match_score', 0.0),
                value_score=p.get('value_score', 0.0),
                heuristic_score=p.get('heuristic_score', 0.0),
            )
            product_models.append(prod_model)

        best_pick = product_models[0] if product_models else None

        return SearchResult(products=product_models, best_pick=best_pick, ai_summary="", total_found=len(products))


__all__ = ['Recommender']
