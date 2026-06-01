"""
Model Training Script for AI Shopping Assistant

This script trains all ML models needed for the recommendation pipeline:
1. TF-IDF vectorizer - for product matching
2. VADER sentiment analyzer - for review sentiment
3. Value score calculator - for rating/price optimization
4. Heuristic score calibrator - for ranking

Run this once during setup to generate trained models and statistics.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon for sentiment analysis
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


class ModelTrainer:
    """Trains and saves all ML models for the shopping assistant."""
    
    def __init__(self, dataset_path: str, output_dir: str = "backend/trained_models"):
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.df = None
        self.tfidf = None
        self.tfidf_matrix = None
        self.sia = SentimentIntensityAnalyzer()
        self.stats = {}
        
    def load_dataset(self):
        """Load Amazon dataset."""
        print("Loading dataset...")
        try:
            self.df = pd.read_csv(self.dataset_path)
            print(f"Dataset loaded: {self.df.shape[0]} products, {self.df.shape[1]} columns")
            print(f"Columns: {self.df.columns.tolist()}")
            return True
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def train_tfidf(self):
        """Train TF-IDF vectorizer on product names and descriptions."""
        print("\n1. Training TF-IDF vectorizer...")
        
        # Combine product features
        product_text = (
            self.df.get('product_name', self.df.get('name', '')).astype(str) + " " +
            self.df.get('category', '').astype(str) + " " +
            self.df.get('about_product', self.df.get('description', '')).astype(str)
        ).str.lower()
        
        # Train TF-IDF
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9
        )
        self.tfidf_matrix = self.tfidf.fit_transform(product_text)
        
        # Save model
        model_path = self.output_dir / "tfidf_vectorizer.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.tfidf, f)
        print(f"   TF-IDF trained: {self.tfidf_matrix.shape}")
        print(f"   Vocabulary size: {len(self.tfidf.vocabulary_)}")
        print(f"   Model saved: {model_path}")
        
        self.stats['tfidf_features'] = len(self.tfidf.vocabulary_)
    
    def calculate_sentiment_scores(self):
        """Calculate sentiment scores for all product reviews."""
        print("\n2. Calculating sentiment scores...")
        
        sentiment_scores = []
        review_col = None
        
        # Find review column
        for col in ['reviews', 'review', 'product_review', 'review_text']:
            if col in self.df.columns:
                review_col = col
                break
        
        if review_col is None:
            print("   No review column found, using default sentiment (0)")
            self.df['sentiment_score'] = 0.0
            return
        
        # Calculate sentiment for each review
        for review in self.df[review_col]:
            if pd.isna(review) or review == '':
                sentiment_scores.append(0.0)
            else:
                scores = self.sia.polarity_scores(str(review))
                sentiment_scores.append(scores['compound'])  # -1 to 1
        
        self.df['sentiment_score'] = sentiment_scores
        
        print(f"   Sentiment scores calculated")
        print(f"   Mean: {np.mean(sentiment_scores):.3f}")
        print(f"   Std: {np.std(sentiment_scores):.3f}")
        print(f"   Range: [{np.min(sentiment_scores):.3f}, {np.max(sentiment_scores):.3f}]")
        
        self.stats['sentiment_mean'] = float(np.mean(sentiment_scores))
        self.stats['sentiment_std'] = float(np.std(sentiment_scores))
    
    def calculate_value_scores(self):
        """Calculate value scores (rating/price ratio normalized)."""
        print("\n3. Calculating value scores...")
        
        # Find rating and price columns
        rating_col = None
        price_col = None
        
        for col in ['rating', 'ratings', 'avg_rating']:
            if col in self.df.columns:
                rating_col = col
                break
        
        for col in ['price', 'discounted_price', 'selling_price', 'product_price']:
            if col in self.df.columns:
                price_col = col
                break
        
        if rating_col and price_col:
            # Normalize rating (0-5 to 0-1)
            ratings = self.df[rating_col].fillna(3).astype(float) / 5.0
            
            # Normalize price (inverse: higher price = lower value)
            prices = self.df[price_col].fillna(self.df[price_col].median()).astype(float)
            max_price = prices.max()
            normalized_price = 1 - (prices / max_price)
            
            # Value score = weighted combination
            value_scores = (0.7 * ratings + 0.3 * normalized_price)
            self.df['value_score'] = np.clip(value_scores, 0, 1)
            
            print(f"   Value scores calculated")
            print(f"   Mean: {self.df['value_score'].mean():.3f}")
            print(f"   Range: [{self.df['value_score'].min():.3f}, {self.df['value_score'].max():.3f}]")
            
            self.stats['value_score_mean'] = float(self.df['value_score'].mean())
        else:
            print(f"   Missing columns: rating={rating_col}, price={price_col}")
            self.df['value_score'] = 0.5
    
    def save_statistics(self):
        """Save training statistics."""
        print("\n4. Saving statistics...")
        
        stats_path = self.output_dir / "training_stats.json"
        self.stats['dataset_size'] = len(self.df)
        self.stats['timestamp'] = pd.Timestamp.now().isoformat()
        
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"   Statistics saved: {stats_path}")
    
    def save_processed_dataset(self):
        """Save processed dataset with all scores."""
        print("\n5. Saving processed dataset...")
        
        output_path = self.output_dir / "processed_amazon.csv"
        self.df.to_csv(output_path, index=False)
        print(f"   Processed dataset saved: {output_path}")
        print(f"   Columns: {self.df.columns.tolist()}")
    
    def train_all(self):
        """Execute complete training pipeline."""
        print("=" * 60)
        print("AI Shopping Assistant - Model Training Pipeline")
        print("=" * 60)
        
        if not self.load_dataset():
            return False
        
        self.train_tfidf()
        self.calculate_sentiment_scores()
        self.calculate_value_scores()
        self.save_statistics()
        self.save_processed_dataset()
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"\nGenerated Files:")
        for file in self.output_dir.glob("*"):
            print(f"  ✓ {file.name}")
        
        return True


def main():
    """Main training entry point."""
    dataset_path = r"dataset\amazon.csv"
    
    trainer = ModelTrainer(dataset_path)
    success = trainer.train_all()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
