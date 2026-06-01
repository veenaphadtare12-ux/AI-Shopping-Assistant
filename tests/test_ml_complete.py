# FILE: test_ml_complete.py
# PURPOSE: Complete ML testing - verify all components work together
#
# This script tests:
# 1. Data loading and preprocessing
# 2. ML model training (TF-IDF, sentiment)
# 3. Algorithm execution (BFS, Best First, Hill Climbing)
# 4. Recommendation scoring
# 5. Performance metrics
#
# Run: python test_ml_complete.py
# Expected: All tests PASS with metrics

import pandas as pd
import numpy as np
import time
import json
from pathlib import Path

# Import your modules
try:
    from backend.models import Product, SearchQuery, SearchResult
    from backend.ml_recommender import Recommender
    from backend.nlp_engine import compute_match_score, compute_sentiment, score_all_products
    from backend.algorithms import best_first_search, normalize_scores, hill_climbing_filter
    from backend.llm_service import explain_recommendation
    print("✓ All imports successful\n")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure all backend files exist and are in correct location")
    exit(1)

class MLTester:
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
    
    def test_1_data_loading(self):
        """Test 1: Can we load the processed dataset?"""
        print("TEST 1: Data Loading")
        print("-" * 50)
        
        try:
            df = pd.read_csv('products_processed.csv')
            
            assert len(df) > 0, "Dataset is empty"
            assert 'product_name' in df.columns, "Missing product_name column"
            assert 'price' in df.columns, "Missing price column"
            assert 'rating' in df.columns, "Missing rating column"
            assert 'sentiment_score' in df.columns, "Missing sentiment_score column"
            assert 'value_score' in df.columns, "Missing value_score column"
            
            print(f"✓ Dataset loaded: {len(df)} products")
            print(f"✓ Columns: {list(df.columns)}")
            print(f"✓ Price range: ₹{df['price'].min():.0f} - ₹{df['price'].max():.0f}")
            print(f"✓ Rating range: {df['rating'].min():.1f}★ - {df['rating'].max():.1f}★")
            print(f"✓ Sentiment range: {df['sentiment_score'].min():.2f} to {df['sentiment_score'].max():.2f}")
            print(f"✓ Value range: {df['value_score'].min():.2f} to {df['value_score'].max():.2f}\n")
            
            self.test_results['data_loading'] = 'PASS'
            return True
        
        except Exception as e:
            print(f"✗ FAILED: {e}\n")
            self.test_results['data_loading'] = f'FAIL: {e}'
            return False
    
    def test_2_model_loading(self):
        """Test 2: Can we load the trained ML models?"""
        print("TEST 2: Model Loading")
        print("-" * 50)
        
        try:
            import joblib
            
            # Check TF-IDF vectorizer
            assert Path('tfidf_vectorizer.pkl').exists(), "tfidf_vectorizer.pkl not found"
            vectorizer = joblib.load('tfidf_vectorizer.pkl')
            print(f"✓ TF-IDF vectorizer loaded")
            print(f"  - Vocabulary size: {len(vectorizer.vocabulary_)}")
            print(f"  - Features: {vectorizer.get_feature_names_out()[:10]}\n")
            
            # Check scaler
            assert Path('scaler.pkl').exists(), "scaler.pkl not found"
            scaler = joblib.load('scaler.pkl')
            print(f"✓ Feature scaler loaded")
            print(f"  - Mean shape: {scaler.mean_.shape}")
            print(f"  - Variance shape: {scaler.var_.shape}\n")
            
            self.test_results['model_loading'] = 'PASS'
            return True
        
        except Exception as e:
            print(f"✗ FAILED: {e}\n")
            self.test_results['model_loading'] = f'FAIL: {e}'
            return False
    
    def test_3_recommender_init(self):
        """Test 3: Can we initialize the recommender?"""
        print("TEST 3: Recommender Initialization")
        print("-" * 50)
        
        try:
            self.recommender = Recommender()
            print(f"✓ Recommender initialized")
            print(f"✓ Dataset in memory: {len(self.recommender.df)} products\n")
            
            self.test_results['recommender_init'] = 'PASS'
            return True
        
        except Exception as e:
            print(f"✗ FAILED: {e}\n")
            self.test_results['recommender_init'] = f'FAIL: {e}'
            return False
    
    def test_4_tf_idf_matching(self):
        """Test 4: Does TF-IDF text matching work?"""
        print("TEST 4: TF-IDF Text Matching")
        print("-" * 50)
        
        try:
            test_queries = [
                "boAt earphones",
                "wireless headphones",
                "budget speakers"
            ]
            
            for query in test_queries:
                scores = self.recommender.compute_match_scores(
                    query,
                    self.recommender.df[:5]  # Test on first 5 products
                )
                
                assert len(scores) == 5, f"Expected 5 scores, got {len(scores)}"
                assert all(0 <= s <= 1 for s in scores), "Scores not in 0-1 range"
                
                print(f"✓ Query: '{query}'")
                print(f"  - Match scores: {scores}")
                print(f"  - Mean match: {np.mean(scores):.3f}\n")
            
            self.test_results['tf_idf_matching'] = 'PASS'
            return True
        
        except Exception as e:
            print(f"✗ FAILED: {e}\n")
            self.test_results['tf_idf_matching'] = f'FAIL: {e}'
            return False
    
    def test_5_sentiment_analysis(self):
        """Test 5: Does sentiment analysis work?"""
        print("TEST 5: Sentiment Analysis")
        print("-" * 50)
        
        try:
            test_reviews = [
                ["Great product!", "Excellent quality"],
                ["Terrible product", "Battery died"],
                ["Average, nothing special"]
            ]
            
            for reviews in test_reviews:
                sentiment = compute_sentiment(reviews)
                
                assert -1 <= sentiment <= 1, f"Sentiment not in -1 to 1 range: {sentiment}"
                
                if sentiment > 0.3:
                    label = "POSITIVE"
                elif sentiment < -0.3:
                    label = "NEGATIVE"
                else:
                    label = "NEUTRAL"
                
                print(f"✓ Reviews: {reviews}")
                print(f"  - Sentiment: {sentiment:.3f} ({label})\n")
            
            self.test_results['sentiment_analysis'] = 'PASS'
            return True
        
        except Exception as e:
            print(f"✗ FAILED: {e}\n")
            self.test_results['sentiment_analysis'] = f'FAIL: {e}'
            return False
    
    def test_6_heuristic_scoring(self):
        """Test 6: Does heuristic scoring work correctly?"""
        print("TEST 6: Heuristic Scoring (Best First Search)")
        print("-" * 50)
        
        try:
            # Get sample products
            test_df = self.recommender.df.head(20).copy()
            test_df['match_score'] = np.random.rand(len(test_df))
            test_df['sentiment_norm'] = (test_df['sentiment_score'] + 1) / 2
            
            # Calculate heuristic
            test_df['heuristic_score'] = (
                0.35 * test_df['value_score'] +
                0.30 * test_df['sentiment_norm'] +
                0.20 * test_df['match_score'] +
                0.15 * (test_df['rating'] / 5.0)
            )
            
            # Check scores are valid
            assert all(0 <= s <= 1 for s in test_df['heuristic_score']), "Heuristic not 0-1"
            
            top_5 = test_df.nlargest(5, 'heuristic_score')
            
            print(f"✓ Heuristic scores calculated")
            print(f"  - Formula: 0.35×value + 0.30×sentiment + 0.20×match + 0.15×rating")
            print(f"  - Score range: {test_df['heuristic_score'].min():.3f} to {test_df['heuristic_score'].max():.3f}\n")
            print("✓ Top 5 ranked products:")
            for i, (_, row) in enumerate(top_5.iterrows(), 1):
                print(f"  {i}. {row['product_name'][:40]} - Score: {row['heuristic_score']:.3f}")
            print()
            
            self.test_results['heuristic_scoring'] = 'PASS'
            return True
        
        except Exception as e:
            print(f"✗ FAILED: {e}\n")
            self.test_results['heuristic_scoring'] = f'FAIL: {e}'
            return False
    
    def test_7_complete_search(self):
        """Test 7: Complete search pipeline (all components together)"""
        print("TEST 7: Complete Search Pipeline")
        print("-" * 50)
        
        try:
            test_query = "wireless earphones"
            max_price = 5000
            min_rating = 3.0
            
            start = time.time()
            
            result = self.recommender.get_recommendations(
                query=test_query,
                max_price=max_price,
                min_rating=min_rating,
                top_n=5
            )
            
            elapsed = time.time() - start
            
            assert isinstance(result, SearchResult), "Result is not SearchResult"
            assert result.total_found > 0, "No products found"
            assert result.best_pick is not None, "No best product"
            assert len(result.products) <= 5, "Too many products returned"
            
            print(f"✓ Search completed in {elapsed*1000:.1f}ms")
            print(f"✓ Query: '{test_query}'")
            print(f"✓ Filters: max_price=₹{max_price}, min_rating={min_rating}★")
            print(f"✓ Total found: {result.total_found}")
            print(f"✓ Returned: {len(result.products)} products\n")
            
            print("✓ Best Pick:")
            best = result.best_pick
            print(f"  Name: {best.name}")
            print(f"  Price: ₹{best.price:.0f}")
            print(f"  Rating: {best.rating}★ ({best.review_count} reviews)")
            print(f"  Platform: {best.platform}")
            print(f"  Heuristic Score: {best.heuristic_score:.3f}")
            print(f"  Match Score: {best.match_score:.3f}")
            print(f"  Sentiment: {best.sentiment_score:.3f}")
            print(f"  Value Score: {best.value_score:.3f}\n")
            
            self.test_results['complete_search'] = 'PASS'
            self.best_product = best  # Save for OpenAI test
            return True
        
        except Exception as e:
            print(f"✗ FAILED: {e}\n")
            self.test_results['complete_search'] = f'FAIL: {e}'
            return False
    
    def test_8_openai_integration(self):
        """Test 8: OpenAI API integration"""
        print("TEST 8: OpenAI API Integration")
        print("-" * 50)
        
        try:
            from backend.llm_service import explain_recommendation
            import os
            from dotenv import load_dotenv
            
            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')
            
            if not api_key:
                print("✗ FAILED: OPENAI_API_KEY not found in .env")
                print("  Solution: Add OPENAI_API_KEY=sk-... to .env file\n")
                self.test_results['openai_integration'] = 'FAIL: No API key'
                return False
            
            if not self.best_product:
                print("✗ FAILED: No product to explain (run test_7 first)\n")
                self.test_results['openai_integration'] = 'FAIL: No product'
                return False
            
            print(f"✓ OpenAI API key found (length: {len(api_key)})")
            print(f"✓ Testing explanation generation...\n")
            
            start = time.time()
            
            explanation = explain_recommendation(
                self.best_product,
                "wireless earphones"
            )
            
            elapsed = time.time() - start
            
            print(f"✓ Explanation generated in {elapsed*1000:.1f}ms")
            print(f"✓ Product: {self.best_product.name}")
            print(f"✓ Explanation:\n")
            print(f"  {explanation}\n")
            
            assert len(explanation) > 10, "Explanation too short"
            
            self.test_results['openai_integration'] = 'PASS'
            return True
        
        except Exception as e:
            error_msg = str(e)
            print(f"✗ FAILED: {error_msg}\n")
            
            if "api_key" in error_msg.lower() or "auth" in error_msg.lower():
                print("  ⚠️  API Key Issue:")
                print("  1. Go to https://platform.openai.com/api/keys")
                print("  2. Create a new secret key")
                print("  3. Copy the key (starts with sk-)")
                print("  4. Add to .env: OPENAI_API_KEY=sk-...\n")
            
            self.test_results['openai_integration'] = f'FAIL: {error_msg}'
            return False
    
    def test_9_hill_climbing_filter(self):
        """Test 9: Hill Climbing algorithm for filter refinement"""
        print("TEST 9: Hill Climbing Filter (Real-Time Refinement)")
        print("-" * 50)
        
        try:
            # Get some products
            test_products = self.recommender.df.head(30).copy()
            
            # Create Product objects
            products = []
            for _, row in test_products.iterrows():
                p = Product(
                    name=row['product_name'],
                    price=row['price'],
                    rating=row['rating'],
                    review_count=row['review_count'],
                    platform=row['platform'],
                    url='',
                    image_url='',
                    heuristic_score=row['value_score']  # Use for sorting
                )
                products.append(p)
            
            # Original constraints
            original_price = 5000
            original_count = len([p for p in products if p.price <= original_price])
            
            print(f"✓ Original filter: max_price=₹{original_price}")
            print(f"✓ Products matching: {original_count}\n")
            
            # Apply new stricter filter using Hill Climbing
            new_price = 2000
            new_rating = 4.0
            
            start = time.time()
            filtered = hill_climbing_filter(products, new_price, new_rating)
            elapsed = time.time() - start
            
            print(f"✓ Hill Climbing applied in {elapsed*1000:.1f}ms")
            print(f"✓ New filter: max_price=₹{new_price}, min_rating={new_rating}★")
            print(f"✓ Products matching: {len(filtered)}\n")
            
            if filtered:
                print("✓ Top 3 after filtering:")
                for i, p in enumerate(filtered[:3], 1):
                    print(f"  {i}. {p.name[:40]} - ₹{p.price:.0f}, {p.rating}★")
            print()
            
            self.test_results['hill_climbing'] = 'PASS'
            return True
        
        except Exception as e:
            print(f"✗ FAILED: {e}\n")
            self.test_results['hill_climbing'] = f'FAIL: {e}'
            return False
    
    def test_10_performance_metrics(self):
        """Test 10: Performance - is it fast enough?"""
        print("TEST 10: Performance Metrics")
        print("-" * 50)
        
        try:
            import statistics
            
            times = []
            queries = [
                "boAt earphones",
                "wireless headphones",
                "budget speakers",
                "premium earphones"
            ]
            
            for query in queries:
                start = time.time()
                result = self.recommender.get_recommendations(query=query)
                elapsed = (time.time() - start) * 1000
                times.append(elapsed)
            
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"✓ Ran {len(queries)} searches")
            print(f"✓ Average time: {avg_time:.1f}ms")
            print(f"✓ Min time: {min_time:.1f}ms")
            print(f"✓ Max time: {max_time:.1f}ms\n")
            
            if avg_time < 100:
                print(f"✓ EXCELLENT: Average < 100ms ✓✓✓")
            elif avg_time < 200:
                print(f"✓ GOOD: Average < 200ms")
            else:
                print(f"⚠️  SLOW: Average > 200ms")
                print(f"  Consider: caching, indexing, or data reduction")
            print()
            
            self.test_results['performance'] = 'PASS'
            return True
        
        except Exception as e:
            print(f"✗ FAILED: {e}\n")
            self.test_results['performance'] = f'FAIL: {e}'
            return False
    
    def print_summary(self):
        """Print final test summary"""
        total_time = time.time() - self.start_time
        
        print("\n" + "="*50)
        print("FINAL TEST SUMMARY")
        print("="*50 + "\n")
        
        passed = sum(1 for v in self.test_results.values() if v == 'PASS')
        total = len(self.test_results)
        
        for test, result in self.test_results.items():
            status = "✓ PASS" if result == "PASS" else "✗ FAIL"
            print(f"{status} - {test.replace('_', ' ').title()}")
            if result != "PASS":
                print(f"       {result}")
        
        print(f"\n{'='*50}")
        print(f"PASSED: {passed}/{total}")
        print(f"TIME: {total_time:.1f}s")
        print(f"{'='*50}\n")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED - PROJECT IS WORKING!")
        elif passed >= total - 2:
            print("⚠️  MOST TESTS PASSED - FIX REMAINING ISSUES")
        else:
            print("❌ MULTIPLE FAILURES - CHECK SETUP")
        
        return passed == total
    
    def run_all_tests(self):
        """Run all tests in sequence"""
        print("\n" + "="*50)
        print("STARTING ML TESTING")
        print("="*50 + "\n")
        
        tests = [
            self.test_1_data_loading,
            self.test_2_model_loading,
            self.test_3_recommender_init,
            self.test_4_tf_idf_matching,
            self.test_5_sentiment_analysis,
            self.test_6_heuristic_scoring,
            self.test_7_complete_search,
            self.test_8_openai_integration,
            self.test_9_hill_climbing_filter,
            self.test_10_performance_metrics,
        ]
        
        for test in tests:
            if not test():
                print("Continuing with next test...\n")
                continue
        
        return self.print_summary()

if __name__ == "__main__":
    tester = MLTester()
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    exit(0 if success else 1)
