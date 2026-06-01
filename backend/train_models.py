"""
Training Pipeline - Modular ML Model Trainer

Trains TF-IDF, sentiment analysis, and scoring models for the shopping assistant.
This handles the Amazon product dataset with columns:
  - product_name, category, about_product (for TF-IDF)
  - review_content (for sentiment)
  - rating, discounted_price (for value scores)

Run: python backend/train_models.py
Output: backend/trained_models/ directory with pkl and csv files
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler


class TFIDFTrainer:
    """Trains and saves TF-IDF vectorizer."""
    
    def __init__(self, output_dir="backend/trained_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.vectorizer = None
        
    def train(self, texts):
        """Train TF-IDF on product descriptions."""
        print("Training TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9
        )
        matrix = self.vectorizer.fit_transform(texts)
        
        # Save
        model_path = self.output_dir / "tfidf_vectorizer.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print(f"  [OK] TF-IDF trained: {matrix.shape}")
        print(f"  [OK] Vocabulary: {len(self.vectorizer.vocabulary_)} terms")
        print(f"  [OK] Saved: {model_path}")
        return matrix


class SentimentScorer:
    """Simple sentiment scorer based on review keywords."""
    
    POSITIVE_WORDS = {
        'good', 'great', 'excellent', 'amazing', 'awesome', 'love', 'perfect',
        'best', 'quality', 'satisfied', 'happy', 'fantastic', 'wonderful',
        'superb', 'outstanding', 'brilliant', 'valuable', 'worthy'
    }
    
    NEGATIVE_WORDS = {
        'bad', 'poor', 'terrible', 'awful', 'hate', 'worst', 'worst',
        'disappointed', 'useless', 'broken', 'waste', 'horrible', 'garbage',
        'junk', 'defective', 'unreliable', 'scam', 'false'
    }
    
    @classmethod
    def score(cls, text):
        """Calculate sentiment score from -1 to 1."""
        if not text or pd.isna(text):
            return 0.0
        
        text_lower = str(text).lower()
        words = text_lower.split()
        
        positive_count = sum(1 for w in words if w in cls.POSITIVE_WORDS)
        negative_count = sum(1 for w in words if w in cls.NEGATIVE_WORDS)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        score = (positive_count - negative_count) / total
        return np.clip(score, -1, 1)


class ValueScoreCalculator:
    """Calculates value scores based on rating and price."""
    
    @staticmethod
    def calculate(ratings, prices):
        """Calculate normalized value scores."""
        # Normalize ratings (0-5 to 0-1)
        ratings_norm = np.clip(ratings / 5.0, 0, 1)
        
        # Normalize prices inversely (lower price = higher value)
        prices = np.array(prices)
        max_price = np.nanmax(prices)
        prices_norm = 1 - (prices / max_price)
        
        # Weighted combination: 70% rating, 30% price
        value_scores = (0.7 * ratings_norm) + (0.3 * prices_norm)
        return np.clip(value_scores, 0, 1)


class ModelTrainer:
    """Complete training pipeline."""
    
    def __init__(self, dataset_path, output_dir="backend/trained_models"):
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.df = None
        self.stats = {}
        
    def load_data(self):
        """Load and explore dataset."""
        print("Loading dataset...")
        try:
            self.df = pd.read_csv(self.dataset_path)
            print(f"  [OK] Loaded: {len(self.df)} products")
            print(f"  [OK] Columns: {', '.join(self.df.columns.tolist()[:5])}...")
            return True
        except Exception as e:
            print(f"  [ERROR] {e}")
            return False
    
    def prepare_text_features(self):
        """Prepare concatenated text features from dataset."""
        print("\nPreparing text features...")
        
        # Our Amazon dataset has these text columns:
        # 'product_name', 'category', 'about_product'
        text_cols = []
        for col in ['product_name', 'category', 'about_product']:
            if col in self.df.columns:
                text_cols.append(col)
        
        if not text_cols:
            print(f"  [ERROR] No text columns found!")
            print(f"  [INFO] Available columns: {self.df.columns.tolist()}")
            return None
        
        # Concatenate and lowercase all text
        features = []
        for col in text_cols:
            features.append(self.df[col].fillna('').astype(str).str.lower())
        
        # Join features with space and convert to list
        combined_text = (features[0] if len(features) >= 1 else '') + ' ' + \
                       (features[1] if len(features) >= 2 else '') + ' ' + \
                       (features[2] if len(features) >= 3 else '')
        
        combined_text = combined_text.str.replace(r'\s+', ' ', regex=True)  # Clean spaces
        
        print(f"  [OK] Combined text from columns: {text_cols}")
        print(f"  [OK] Total documents: {len(combined_text)}")
        return combined_text.tolist()
    
    def train_tfidf(self, texts):
        """Train TF-IDF vectorizer."""
        print("\n1. TF-IDF Training")
        print("-" * 40)
        trainer = TFIDFTrainer(self.output_dir)
        tfidf_matrix = trainer.train(texts)
        self.stats['tfidf_vocab_size'] = len(trainer.vectorizer.vocabulary_)
        return tfidf_matrix
    
    def calculate_sentiment(self):
        """Calculate sentiment scores from review_content."""
        print("\n2. Sentiment Scoring")
        print("-" * 40)
        
        # Amazon dataset uses 'review_content' for reviews
        review_col = None
        for col in ['review_content', 'reviews', 'review', 'product_review']:
            if col in self.df.columns:
                review_col = col
                break
        
        if review_col:
            print(f"  [OK] Using '{review_col}' column for sentiment")
            sentiments = self.df[review_col].apply(SentimentScorer.score)
        else:
            print(f"  [WARN] No review column found. Available: {self.df.columns.tolist()}")
            print(f"  [WARN] Using default sentiment (0)")
            sentiments = pd.Series([0.0] * len(self.df))
        
        self.df['sentiment_score'] = sentiments
        mean_sent = sentiments.mean()
        min_sent = sentiments.min()
        max_sent = sentiments.max()
        
        print(f"  [OK] Mean sentiment: {mean_sent:.3f}")
        print(f"  [OK] Range: [{min_sent:.3f}, {max_sent:.3f}]")
        
        self.stats['sentiment_mean'] = float(mean_sent)
        self.stats['sentiment_range'] = [float(min_sent), float(max_sent)]
    
    def calculate_value_scores(self):
        """Calculate value scores from Amazon dataset columns."""
        print("\n3. Value Score Calculation")
        print("-" * 40)
        
        # Amazon dataset columns: 'rating', 'discounted_price'
        rating_col = None
        price_col = None
        
        for col in ['rating', 'ratings', 'avg_rating', 'product_rating']:
            if col in self.df.columns:
                rating_col = col
                break
        
        for col in ['discounted_price', 'price', 'selling_price', 'product_price']:
            if col in self.df.columns:
                price_col = col
                break
        
        if rating_col and price_col:
            # Parse rating
            ratings = pd.to_numeric(self.df[rating_col], errors='coerce').fillna(3).astype(float)
            
            # Parse price - handle currency strings like "₹399" or "₹1,099"
            def parse_price(price_str):
                if pd.isna(price_str):
                    return None
                price_str = str(price_str)
                # Remove currency symbols and commas
                price_str = price_str.replace('₹', '').replace(',', '').strip()
                try:
                    return float(price_str)
                except ValueError:
                    return None
            
            prices = self.df[price_col].apply(parse_price)
            prices = prices.fillna(prices.median())
            
            value_scores = ValueScoreCalculator.calculate(ratings, prices)
            self.df['value_score'] = value_scores
            
            print(f"  [OK] Rating column: {rating_col}")
            print(f"  [OK] Price column: {price_col}")
            print(f"  [OK] Mean value score: {value_scores.mean():.3f}")
            print(f"  [OK] Rating range: [{ratings.min():.2f}, {ratings.max():.2f}]")
            print(f"  [OK] Price range: [{prices.min():.0f}, {prices.max():.0f}]")
            
            self.stats['value_score_mean'] = float(value_scores.mean())
            self.stats['rating_range'] = [float(ratings.min()), float(ratings.max())]
            self.stats['price_range'] = [float(prices.min()), float(prices.max())]
        else:
            print(f"  [WARN] Missing columns: rating={rating_col}, price={price_col}")
            print(f"  [WARN] Available: {self.df.columns.tolist()}")
            self.df['value_score'] = 0.5
    
    def save_outputs(self):
        """Save processed dataset and statistics."""
        print("\n4. Saving Outputs")
        print("-" * 40)
        
        # Save processed dataset
        output_path = self.output_dir / "processed_data.csv"
        self.df.to_csv(output_path, index=False)
        output_size = output_path.stat().st_size / (1024 * 1024)
        print(f"  [OK] Dataset saved: {output_path} ({output_size:.1f} MB)")
        
        # Save statistics
        self.stats['dataset_size'] = len(self.df)
        self.stats['columns_in_output'] = self.df.columns.tolist()
        self.stats['timestamp'] = pd.Timestamp.now().isoformat()
        
        stats_path = self.output_dir / "training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"  [OK] Statistics saved: {stats_path}")
    
    def train_all(self):
        """Execute complete training pipeline."""
        print("\n" + "=" * 50)
        print("AI Shopping Assistant - Model Training Pipeline")
        print("=" * 50)
        
        if not self.load_data():
            return False
        
        # Prepare and train
        texts = self.prepare_text_features()
        self.train_tfidf(texts)
        self.calculate_sentiment()
        self.calculate_value_scores()
        self.save_outputs()
        
        # Summary
        print("\n" + "=" * 50)
        print("Training Complete! Files Generated:")
        print("=" * 50)
        for file in sorted(self.output_dir.glob("*")):
            size = file.stat().st_size
            if size > 1024*1024:
                size_str = f"{size/(1024*1024):.1f}MB"
            else:
                size_str = f"{size/1024:.1f}KB"
            print(f"  [OK] {file.name:40} ({size_str})")
        
        return True


def main():
    """Main entry point."""
    import sys
    
    # Check if dataset path provided
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else r"dataset\amazon.csv"
    
    trainer = ModelTrainer(dataset_path)
    success = trainer.train_all()
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    exit(main())
