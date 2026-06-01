"""
Unified Model Training Script for AI Shopping Assistant.

Trains ML features on combined data from:
1. Amazon dataset
2. Myntra dataset
3. Flipkart dataset

The script standardizes each source into one schema, trains a TF-IDF
vectorizer, computes sentiment/value scores, and saves the merged dataset
plus training statistics.
"""

import json
import pickle
import re
import warnings
from pathlib import Path
from typing import Dict, Optional

import nltk
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

# Download VADER lexicon for sentiment analysis
try:
    nltk.data.find("vader_lexicon")
except LookupError:
    try:
        nltk.download("vader_lexicon", quiet=True)
    except Exception:
        pass


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AMAZON_PATH = PROJECT_ROOT / "dataset" / "amazon.csv" / "amazon.csv"
DEFAULT_MYNTRA_PATH = Path.home() / "OneDrive" / "ドキュメント" / "Downloads" / "archive (1)" / "myntra202305041052.csv"
DEFAULT_FLIPKART_PATH = Path.home() / "OneDrive" / "ドキュメント" / "Downloads" / "archive (2)" / "train.csv"


def parse_numeric(value, default: float = 0.0) -> float:
    """Parse prices/counts stored as text like '₹1,299' or '24,269'."""
    if pd.isna(value):
        return default

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    if not text:
        return default

    cleaned = re.sub(r"[^0-9.\-]", "", text)
    if cleaned in {"", "-", ".", "-."}:
        return default

    try:
        return float(cleaned)
    except ValueError:
        return default


def resolve_dataset_path(label: str, env_name: str, default_path: Path) -> Optional[Path]:
    """Resolve dataset path from env var first, otherwise from default path."""
    import os

    env_path = os.getenv(env_name)
    candidate = Path(env_path).expanduser() if env_path else default_path

    if candidate.exists():
        print(f"  [OK] {label} dataset found")
        return candidate

    print(f"  [WARN] {label} dataset not found")
    return None


class SimpleSentimentScorer:
    """Offline fallback sentiment scorer for environments without VADER data."""

    POSITIVE_WORDS = {
        "good", "great", "excellent", "amazing", "love", "best", "quality",
        "comfortable", "fast", "value", "nice", "solid", "smooth", "happy",
    }

    NEGATIVE_WORDS = {
        "bad", "poor", "worst", "slow", "issue", "problem", "broken",
        "average", "cheap", "refund", "defect", "disappointed", "weak",
    }

    @classmethod
    def score(cls, text: str) -> float:
        """Return a crude compound score in [-1, 1]."""
        if pd.isna(text) or not str(text).strip():
            return 0.0

        words = re.findall(r"[a-zA-Z]+", str(text).lower())
        if not words:
            return 0.0

        positive = sum(1 for word in words if word in cls.POSITIVE_WORDS)
        negative = sum(1 for word in words if word in cls.NEGATIVE_WORDS)
        total = positive + negative
        if total == 0:
            return 0.0
        return max(-1.0, min(1.0, (positive - negative) / total))


class UnifiedModelTrainer:
    """Trains ML models on combined datasets from multiple platforms."""

    def __init__(self, output_dir: str = "backend/trained_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.df_combined = None
        self.tfidf = None
        self.tfidf_matrix = None
        self.sia = None
        self.use_vader = False
        try:
            self.sia = SentimentIntensityAnalyzer()
            self.use_vader = True
        except LookupError:
            print("[WARN] VADER lexicon unavailable. Falling back to keyword-based sentiment scoring.")
        self.stats: Dict[str, object] = {}
        self.dataset_stats: Dict[str, Dict[str, object]] = {}

    def load_amazon_dataset(self, path: Path) -> pd.DataFrame:
        """Load and standardize the Amazon dataset."""
        print("\n" + "=" * 60)
        print("Loading Amazon Dataset...")
        print("=" * 60)

        try:
            df = pd.read_csv(path)
            print(f"[OK] Loaded: {df.shape[0]} products")

            df["product_id"] = df.get("product_id", df.index).astype(str)
            df["product_name"] = df.get("product_name", "").astype(str)
            df["category"] = df.get("category", "").astype(str)
            df["price"] = df.get("discounted_price", 0).apply(parse_numeric)
            df["mrp"] = df.get("actual_price", 0).apply(parse_numeric)
            df["rating"] = df.get("rating", 0).apply(parse_numeric)
            df["rating_count"] = df.get("rating_count", 0).apply(lambda v: int(parse_numeric(v, 0)))
            df["description"] = df.get("about_product", "").astype(str)
            df["url"] = df.get("product_link", "").astype(str)
            df["image_url"] = df.get("img_link", "").astype(str)
            df["platform"] = "Amazon"
            df["seller"] = ""

            final_cols = [
                "product_id",
                "product_name",
                "category",
                "price",
                "mrp",
                "rating",
                "rating_count",
                "description",
                "url",
                "image_url",
                "platform",
                "seller",
            ]
            df = df[final_cols].copy()
            df = df[df["product_name"].str.strip() != ""]
            df = df[df["price"] > 0]

            self.dataset_stats["amazon"] = {
                "count": int(len(df)),
                "avg_rating": float(df["rating"].mean()) if not df.empty else 0.0,
                "avg_price": float(df["price"].mean()) if not df.empty else 0.0,
            }
            return df

        except Exception as exc:
            print(f"[ERROR] Amazon load failed: {exc}")
            return pd.DataFrame()

    def load_myntra_dataset(self, path: Path, sample_size: int = 5000) -> pd.DataFrame:
        """Load and standardize the Myntra dataset."""
        print("\n" + "=" * 60)
        print(f"Loading Myntra Dataset (sampling {sample_size} rows)...")
        print("=" * 60)

        try:
            df = pd.read_csv(path, nrows=sample_size)
            print(f"[OK] Loaded: {len(df)} products")

            df["product_id"] = "myntra_" + df.get("id", df.index).astype(str)
            df["product_name"] = df.get("name", "").astype(str)
            df["category"] = "Fashion"
            df["price"] = df.get("price", 0).apply(parse_numeric)
            df["mrp"] = df.get("mrp", 0).apply(parse_numeric)
            df["rating"] = df.get("rating", 0).apply(parse_numeric)
            df["rating_count"] = df.get("ratingTotal", 0).apply(lambda v: int(parse_numeric(v, 0)))
            df["description"] = df.get("seller", "").astype(str)
            df["url"] = df.get("purl", "").astype(str)
            df["image_url"] = df.get("img", "").astype(str)
            df["platform"] = "Myntra"
            df["seller"] = df.get("seller", "").astype(str)

            final_cols = [
                "product_id",
                "product_name",
                "category",
                "price",
                "mrp",
                "rating",
                "rating_count",
                "description",
                "url",
                "image_url",
                "platform",
                "seller",
            ]
            df = df[final_cols].copy()
            df = df[df["product_name"].str.strip() != ""]
            df = df[df["price"] > 0]

            self.dataset_stats["myntra"] = {
                "count": int(len(df)),
                "avg_rating": float(df["rating"].mean()) if not df.empty else 0.0,
                "avg_price": float(df["price"].mean()) if not df.empty else 0.0,
            }
            return df

        except Exception as exc:
            print(f"[ERROR] Myntra load failed: {exc}")
            return pd.DataFrame()

    def load_flipkart_dataset(self, path: Path) -> pd.DataFrame:
        """Load and standardize the Flipkart dataset."""
        print("\n" + "=" * 60)
        print("Loading Flipkart Dataset...")
        print("=" * 60)

        try:
            df = pd.read_csv(path)
            print(f"[OK] Loaded: {df.shape[0]} products")

            df["product_id"] = "flipkart_" + df.get("id", df.index).astype(str)
            df["product_name"] = df.get("title", "").astype(str)
            df["category"] = df.get("maincateg", "").astype(str)
            df["price"] = df.get("price1", 0).apply(parse_numeric)
            df["mrp"] = df.get("actprice1", 0).apply(parse_numeric)
            df["rating"] = df.get("Rating", 0).apply(parse_numeric)
            df["rating_count"] = df.get("noreviews1", 0).apply(lambda v: int(parse_numeric(v, 0)))
            df["description"] = ""
            df["url"] = ""
            df["image_url"] = ""
            df["platform"] = "Flipkart"
            df["seller"] = ""

            final_cols = [
                "product_id",
                "product_name",
                "category",
                "price",
                "mrp",
                "rating",
                "rating_count",
                "description",
                "url",
                "image_url",
                "platform",
                "seller",
            ]
            df = df[final_cols].copy()
            df = df[df["product_name"].str.strip() != ""]
            df = df[df["price"] > 0]

            self.dataset_stats["flipkart"] = {
                "count": int(len(df)),
                "avg_rating": float(df["rating"].mean()) if not df.empty else 0.0,
                "avg_price": float(df["price"].mean()) if not df.empty else 0.0,
            }
            return df

        except Exception as exc:
            print(f"[ERROR] Flipkart load failed: {exc}")
            return pd.DataFrame()

    def combine_datasets(self, dataset_paths: Dict[str, Optional[Path]], myntra_sample_size: int = 5000):
        """Combine all available datasets into a unified table."""
        print("\n" + "=" * 60)
        print("COMBINING DATASETS")
        print("=" * 60)

        dfs = []

        if dataset_paths.get("amazon"):
            df_amazon = self.load_amazon_dataset(dataset_paths["amazon"])
            if not df_amazon.empty:
                dfs.append(df_amazon)

        if dataset_paths.get("myntra"):
            df_myntra = self.load_myntra_dataset(dataset_paths["myntra"], sample_size=myntra_sample_size)
            if not df_myntra.empty:
                dfs.append(df_myntra)

        if dataset_paths.get("flipkart"):
            df_flipkart = self.load_flipkart_dataset(dataset_paths["flipkart"])
            if not df_flipkart.empty:
                dfs.append(df_flipkart)

        if not dfs:
            raise FileNotFoundError("No dataset files could be loaded.")

        self.df_combined = pd.concat(dfs, ignore_index=True)
        self.df_combined = self.df_combined[self.df_combined["product_name"].str.strip() != ""].copy()

        print("\n" + "-" * 60)
        print(f"[OK] Combined dataset size: {len(self.df_combined)} products")
        print(f"  - Platforms: {self.df_combined['platform'].value_counts().to_dict()}")
        print(f"  - Avg rating: {self.df_combined['rating'].mean():.2f}")
        print(f"  - Avg price: Rs {self.df_combined['price'].mean():.2f}")
        print(f"  - Products with ratings: {(self.df_combined['rating'] > 0).sum()}")
        return self.df_combined

    def train_tfidf(self):
        """Train TF-IDF vectorizer on product names and descriptions."""
        print("\n" + "=" * 60)
        print("TRAINING TF-IDF VECTORIZER")
        print("=" * 60)

        product_text = (
            self.df_combined["product_name"].fillna("").astype(str) + " " +
            self.df_combined["category"].fillna("").astype(str) + " " +
            self.df_combined["description"].fillna("").astype(str)
        ).str.lower()
        product_text = product_text[product_text.str.len() > 0]

        print(f"Texts for TF-IDF: {len(product_text)}")

        self.tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
        )
        self.tfidf_matrix = self.tfidf.fit_transform(product_text)

        model_path = self.output_dir / "tfidf_vectorizer.pkl"
        with open(model_path, "wb") as file_handle:
            pickle.dump(self.tfidf, file_handle)

        print("[OK] TF-IDF trained successfully")
        print(f"  - Feature matrix shape: {self.tfidf_matrix.shape}")
        print(f"  - Vocabulary size: {len(self.tfidf.vocabulary_)}")
        print(f"  - Model saved: {model_path}")

        self.stats["tfidf_features"] = len(self.tfidf.vocabulary_)
        self.stats["tfidf_matrix_shape"] = str(self.tfidf_matrix.shape)

    def calculate_sentiment_scores(self):
        """Calculate sentiment scores for all products."""
        print("\n" + "=" * 60)
        print("CALCULATING SENTIMENT SCORES")
        print("=" * 60)

        sentiment_scores = []
        texts = (
            self.df_combined["description"].fillna("").astype(str) + " " +
            self.df_combined["product_name"].fillna("").astype(str)
        )

        for index, text in enumerate(texts):
            if index % 5000 == 0:
                print(f"  Processing: {index}/{len(texts)}")

            if pd.isna(text) or not str(text).strip():
                sentiment_scores.append(0.0)
            else:
                if self.use_vader:
                    scores = self.sia.polarity_scores(str(text))
                    sentiment_scores.append(scores["compound"])
                else:
                    sentiment_scores.append(SimpleSentimentScorer.score(str(text)))

        self.df_combined["sentiment_score"] = sentiment_scores

        print("\n[OK] Sentiment scores calculated")
        print(f"  - Mean: {np.mean(sentiment_scores):.3f}")
        print(f"  - Std: {np.std(sentiment_scores):.3f}")
        print(f"  - Range: [{np.min(sentiment_scores):.3f}, {np.max(sentiment_scores):.3f}]")

        self.stats["sentiment_mean"] = float(np.mean(sentiment_scores))
        self.stats["sentiment_std"] = float(np.std(sentiment_scores))

    def calculate_value_scores(self):
        """Calculate value scores from rating and price."""
        print("\n" + "=" * 60)
        print("CALCULATING VALUE SCORES")
        print("=" * 60)

        ratings = self.df_combined["rating"].fillna(3).astype(float) / 5.0
        prices = self.df_combined["price"].fillna(self.df_combined["price"].median()).astype(float)
        prices = prices.replace(0, self.df_combined["price"].median())
        max_price = prices.max()

        if max_price > 0:
            normalized_price = 1 - (prices / max_price)
        else:
            normalized_price = 0.5

        value_scores = (0.7 * ratings) + (0.3 * normalized_price)
        self.df_combined["value_score"] = np.clip(value_scores, 0, 1)

        print("[OK] Value scores calculated")
        print(f"  - Mean: {value_scores.mean():.3f}")
        print("  - Range: [0, 1]")

        self.stats["value_score_mean"] = float(value_scores.mean())

    def save_results(self):
        """Save processed data and training statistics."""
        print("\n" + "=" * 60)
        print("SAVING RESULTS")
        print("=" * 60)

        data_path = self.output_dir / "processed_data_combined.csv"
        self.df_combined.to_csv(data_path, index=False)
        print(f"[OK] Combined data saved: {data_path}")

        stats_path = self.output_dir / "training_stats.json"
        stats_to_save = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "total_products": int(len(self.df_combined)),
            "dataset_stats": self.dataset_stats,
            "model_stats": self.stats,
        }
        with open(stats_path, "w", encoding="utf-8") as file_handle:
            json.dump(stats_to_save, file_handle, indent=2)
        print(f"[OK] Statistics saved: {stats_path}")

        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print("\nDatasets Combined:")
        for platform, info in self.dataset_stats.items():
            print(f"  {platform.capitalize()}: {info['count']} products")

        print(f"\nTotal Products: {len(self.df_combined)}")
        print("\nTrained Models:")
        print(f"  [OK] TF-IDF Vectorizer ({self.stats['tfidf_features']} features)")
        print(f"  [OK] Sentiment Scores (mean: {self.stats['sentiment_mean']:.3f})")
        print(f"  [OK] Value Scores (mean: {self.stats['value_score_mean']:.3f})")
        print(f"\nOutput Directory: {self.output_dir}")
        print("=" * 60)


def main():
    """Main training function."""
    print("\n")
    print("=" * 60)
    print("AI SHOPPING ASSISTANT - UNIFIED MODEL TRAINING")
    print("=" * 60)

    dataset_paths = {
        "amazon": resolve_dataset_path("Amazon", "AMAZON_DATASET_PATH", DEFAULT_AMAZON_PATH),
        "myntra": resolve_dataset_path("Myntra", "MYNTRA_DATASET_PATH", DEFAULT_MYNTRA_PATH),
        "flipkart": resolve_dataset_path("Flipkart", "FLIPKART_DATASET_PATH", DEFAULT_FLIPKART_PATH),
    }

    trainer = UnifiedModelTrainer()

    try:
        trainer.combine_datasets(dataset_paths)
        trainer.train_tfidf()
        trainer.calculate_sentiment_scores()
        trainer.calculate_value_scores()
        trainer.save_results()
        print("\n[OK] Training completed successfully!")
    except Exception as exc:
        print(f"\n[ERROR] Error during training: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
