"""Setup test artifacts for local tests:
- copies backend/trained_models/processed_data_combined.csv -> products_processed.csv
- copies tfidf_vectorizer.pkl -> tfidf_vectorizer.pkl (root)
- creates scaler.pkl by fitting StandardScaler on numeric columns (or a dummy fallback)

Run from AI Shopping Assistant folder:
    python setup_test_artifacts.py
"""
from pathlib import Path
import shutil
import sys
import numpy as np

root = Path(__file__).parent
src_models = root / 'backend' / 'trained_models'
src_csv = src_models / 'processed_data_combined.csv'
dst_csv = root / 'products_processed.csv'
src_tfidf = src_models / 'tfidf_vectorizer.pkl'
dst_tfidf = root / 'tfidf_vectorizer.pkl'
dst_scaler = root / 'scaler.pkl'

print(f"Looking for source CSV: {src_csv}")
if not src_csv.exists():
    print("ERROR: source processed CSV not found.")
    sys.exit(1)

shutil.copy2(src_csv, dst_csv)
print(f"Copied {src_csv.name} -> {dst_csv}")

if src_tfidf.exists():
    shutil.copy2(src_tfidf, dst_tfidf)
    print(f"Copied {src_tfidf.name} -> {dst_tfidf}")
else:
    print("Warning: tfidf_vectorizer.pkl not found in trained_models.")

# Create scaler.pkl
try:
    import pandas as pd
    try:
        from sklearn.preprocessing import StandardScaler
        has_sklearn = True
    except Exception:
        has_sklearn = False

    df = pd.read_csv(dst_csv)
    numeric_cols = [c for c in ['price', 'rating', 'sentiment_score', 'value_score'] if c in df.columns]
    if not numeric_cols:
        print("No numeric columns found to fit scaler; creating dummy scaler.")
        has_sklearn = False

    if has_sklearn:
        scaler = StandardScaler()
        X = df[numeric_cols].fillna(0).values
        scaler.fit(X)
        print(f"Fitted StandardScaler on columns: {numeric_cols}")
    else:
        # Fallback dummy scaler
        class DummyScaler:
            pass
        arr = df[numeric_cols].fillna(0).values if numeric_cols else np.zeros((1,1))
        dummy = DummyScaler()
        dummy.mean_ = np.mean(arr, axis=0)
        dummy.var_ = np.var(arr, axis=0)
        scaler = dummy
        print("Created fallback DummyScaler with mean_/var_.")

    # Prefer joblib if available
    try:
        import joblib
        joblib.dump(scaler, dst_scaler)
        print(f"Saved scaler to {dst_scaler} using joblib.")
    except Exception:
        import pickle
        with open(dst_scaler, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Saved scaler to {dst_scaler} using pickle (fallback).")

    print("Setup artifacts complete.")
    sys.exit(0)

except Exception as e:
    print(f"ERROR during setup: {e}")
    sys.exit(1)
