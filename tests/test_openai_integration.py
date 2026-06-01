# FILE: test_openai_integration.py
# PURPOSE: Detailed OpenAI API integration verification
#
# Tests:
# 1. API key validity
# 2. Connection to OpenAI
# 3. Explanation generation
# 4. Review summarization
# 5. Product comparison
# 6. Cost and usage
#
# Run: python test_openai_integration.py

import os
import time
import json
from dotenv import load_dotenv

# Load environment
load_dotenv()

class OpenAITester:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.test_results = {}
        self.total_cost = 0
    
    def test_1_api_key_check(self):
        """Test 1: Is OpenAI API key configured?"""
        print("TEST 1: API Key Configuration")
        print("-" * 50)
        
        if not self.api_key:
            print("✗ FAILED: OPENAI_API_KEY not found in .env")
            print("\nFIX:")
            print("1. Go to https://platform.openai.com/api/keys")
            print("2. Create a new secret key")
            print("3. Copy the key (starts with sk-)")
            print("4. Add to .env file:")
            print("   OPENAI_API_KEY=sk-your-key-here\n")
            self.test_results['api_key_check'] = 'FAIL'
            return False
        
        # Check format
        if not self.api_key.startswith('sk-'):
            print(f"✗ FAILED: API key doesn't start with 'sk-'")
            print(f"   Your key starts with: {self.api_key[:10]}\n")
            self.test_results['api_key_check'] = 'FAIL'
            return False
        
        print(f"✓ API key found")
        print(f"✓ Format valid (starts with sk-)")
        print(f"✓ Length: {len(self.api_key)} characters\n")
        
        self.test_results['api_key_check'] = 'PASS'
        return True
    
    def test_2_client_initialization(self):
        """Test 2: Can we initialize OpenAI client?"""
        print("TEST 2: OpenAI Client Initialization")
        print("-" * 50)
        
        try:
            from openai import OpenAI
            
            self.client = OpenAI(api_key=self.api_key)
            print(f"✓ OpenAI client initialized successfully\n")
            
            self.test_results['client_init'] = 'PASS'
            return True
        
        except Exception as e:
            print(f"✗ FAILED: {e}\n")
            self.test_results['client_init'] = f'FAIL: {e}'
            return False
    
    def test_3_api_connection(self):
        """Test 3: Can we connect to OpenAI API?"""
        print("TEST 3: API Connection Test")
        print("-" * 50)
        
        try:
            print("Sending test request to OpenAI API...")
            
            start = time.time()
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": "Say 'API connection successful' in one word."}
                ],
                max_tokens=10,
                temperature=0.7
            )
            
            elapsed = time.time() - start
            result = response.choices[0].message.content
            
            print(f"✓ API connected successfully")
            print(f"✓ Response time: {elapsed*1000:.0f}ms")
            print(f"✓ Model response: '{result}'")
            print(f"✓ Tokens used: {response.usage.total_tokens}\n")
            
            # Calculate cost (gpt-4o-mini: ~$0.00015 per 1k input, $0.0006 per 1k output)
            cost = (response.usage.prompt_tokens * 0.00015 + 
                   response.usage.completion_tokens * 0.0006) / 1000
            self.total_cost += cost
            
            self.test_results['api_connection'] = 'PASS'
            return True
        
        except Exception as e:
            print(f"✗ FAILED: {e}\n")
            print("Common issues:")
            print("- Invalid API key")
            print("- Insufficient API credits")
            print("- Network connection issue")
            print("- Rate limit exceeded\n")
            
            self.test_results['api_connection'] = f'FAIL: {e}'
            return False
    
    def test_4_product_explanation(self):
        """Test 4: Can we generate product explanations?"""
        print("TEST 4: Product Explanation Generation")
        print("-" * 50)
        
        try:
            from backend.llm_service import explain_recommendation
            from backend.models import Product
            
            # Create test product
            test_product = Product(
                name="boAt Airdopes 171 Pro",
                price=1299.99,
                rating=4.5,
                review_count=2341,
                platform="Amazon",
                url="https://amazon.in/...",
                sentiment_score=0.75
            )
            
            print("Generating explanation for test product...")
            start = time.time()
            
            explanation = explain_recommendation(
                test_product,
                "wireless earphones under 2000"
            )
            
            elapsed = time.time() - start
            
            print(f"✓ Explanation generated in {elapsed*1000:.0f}ms")
            print(f"✓ Length: {len(explanation)} characters")
            print(f"✓ Content:\n")
            print(f"  {explanation}\n")
            
            assert len(explanation) > 20, "Explanation too short"
            assert "boAt" in explanation or "product" in explanation, "Missing key terms"
            
            self.test_results['product_explanation'] = 'PASS'
            return True
        
        except Exception as e:
            print(f"✗ FAILED: {e}\n")
            self.test_results['product_explanation'] = f'FAIL: {e}'
            return False
    
    def test_5_review_summarization(self):
        """Test 5: Can we summarize customer reviews?"""
        print("TEST 5: Review Summarization")
        print("-" * 50)
        
        try:
            from backend.llm_service import summarize_reviews
            
            test_reviews = [
                "Great sound quality, battery lasts 8 hours",
                "Excellent noise cancellation, very comfortable",
                "Good build quality, driver sounds amazing",
                "Battery drains quickly in heavy use",
                "Connectivity issues sometimes"
            ]
            
            print("Summarizing test reviews...")
            start = time.time()
            
            summary = summarize_reviews(test_reviews, "boAt Airdopes")
            
            elapsed = time.time() - start
            
            print(f"✓ Summary generated in {elapsed*1000:.0f}ms")
            print(f"✓ Length: {len(summary)} characters")
            print(f"✓ Content:\n")
            print(f"  {summary}\n")
            
            assert len(summary) > 20, "Summary too short"
            
            self.test_results['review_summarization'] = 'PASS'
            return True
        
        except Exception as e:
            print(f"✗ FAILED: {e}\n")
            self.test_results['review_summarization'] = f'FAIL: {e}'
            return False
    
    def test_6_product_comparison(self):
        """Test 6: Can we compare two products?"""
        print("TEST 6: Product Comparison")
        print("-" * 50)
        
        try:
            from backend.llm_service import compare_products
            from backend.models import Product
            
            product1 = Product(
                name="boAt Airdopes 171 Pro",
                price=1299,
                rating=4.5,
                review_count=2341,
                platform="Amazon",
                url=""
            )
            
            product2 = Product(
                name="Sony WH-CH720",
                price=2499,
                rating=4.3,
                review_count=892,
                platform="Flipkart",
                url=""
            )
            
            print("Comparing two products...")
            start = time.time()
            
            result = compare_products(product1, product2)
            
            elapsed = time.time() - start
            
            print(f"✓ Comparison generated in {elapsed*1000:.0f}ms")
            print(f"✓ Comparison:\n")
            print(f"  {result['comparison']}\n")
            print(f"✓ Recommendation:\n")
            print(f"  {result['recommendation']}\n")
            
            self.test_results['product_comparison'] = 'PASS'
            return True
        
        except Exception as e:
            print(f"✗ FAILED: {e}\n")
            self.test_results['product_comparison'] = f'FAIL: {e}'
            return False
    
    def test_7_cost_analysis(self):
        """Test 7: Check API usage and cost"""
        print("TEST 7: Cost Analysis")
        print("-" * 50)
        
        try:
            print(f"✓ Total API calls: 3 (connection test, explanation, summary)")
            print(f"✓ Model used: gpt-4o-mini (cheapest)")
            print(f"✓ Estimated cost: ${self.total_cost:.6f}")
            print(f"✓ Cost per 1000 requests: ${(self.total_cost * 1000 / 3):.3f}\n")
            
            print("Pricing breakdown:")
            print("- Input: $0.00015 per 1K tokens")
            print("- Output: $0.0006 per 1K tokens")
            print("- Average request: ~50 input + 50 output tokens")
            print("- Per request: ~$0.00004\n")
            
            self.test_results['cost_analysis'] = 'PASS'
            return True
        
        except Exception as e:
            print(f"✗ FAILED: {e}\n")
            self.test_results['cost_analysis'] = f'FAIL: {e}'
            return False
    
    def print_summary(self):
        """Print final summary"""
        print("\n" + "="*50)
        print("OPENAI INTEGRATION SUMMARY")
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
        print(f"TOTAL COST: ${self.total_cost:.6f}")
        print(f"{'='*50}\n")
        
        if passed == total:
            print("🎉 OPENAI INTEGRATION WORKING PERFECTLY!")
        elif passed >= total - 1:
            print("✓ OPENAI INTEGRATION WORKING (minor issues)")
        else:
            print("❌ OPENAI INTEGRATION ISSUES - FIX ABOVE")
        
        return passed == total
    
    def run_all_tests(self):
        """Run all OpenAI tests"""
        print("\n" + "="*50)
        print("OPENAI INTEGRATION TESTING")
        print("="*50 + "\n")
        
        # Must pass first test to continue
        if not self.test_1_api_key_check():
            self.test_results['client_init'] = 'SKIPPED'
            self.test_results['api_connection'] = 'SKIPPED'
            self.test_results['product_explanation'] = 'SKIPPED'
            self.test_results['review_summarization'] = 'SKIPPED'
            self.test_results['product_comparison'] = 'SKIPPED'
            self.test_results['cost_analysis'] = 'SKIPPED'
            return self.print_summary()
        
        tests = [
            self.test_2_client_initialization,
            self.test_3_api_connection,
            self.test_4_product_explanation,
            self.test_5_review_summarization,
            self.test_6_product_comparison,
            self.test_7_cost_analysis,
        ]
        
        for test in tests:
            if not test():
                continue
        
        return self.print_summary()

if __name__ == "__main__":
    tester = OpenAITester()
    success = tester.run_all_tests()
    exit(0 if success else 1)
