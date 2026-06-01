"""
Test script for the AI Shopping Assistant API.

Tests the main endpoints:
- /health
- /search
- /filter
- /compare
"""

import json
import os

import requests

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_health():
    """Test the /health endpoint."""
    print_section("TEST 1: Health Check Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        assert response.status_code == 200
        print("[PASSED] Health check working")
        return True
    except Exception as exc:
        print(f"[FAILED] {exc}")
        return False


def test_search():
    """Test the /search endpoint."""
    print_section("TEST 2: Search Endpoint")
    try:
        payload = {
            "query": "wireless headphones",
            "max_price": 50000,
            "min_rating": 3.0,
            "limit": 10,
            "platforms": ["Amazon", "Flipkart", "Myntra"],
        }
        print(f"Request Payload: {json.dumps(payload, indent=2)}")
        response = requests.post(f"{BASE_URL}/search", json=payload, timeout=20)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Total Products Found: {data.get('total_found')}")
            print(f"Number of Products Returned: {len(data.get('products', []))}")
            if data.get("best_pick"):
                print(f"Best Pick: {data['best_pick'].get('name')}")
                print(f"AI Explanation: {data['best_pick'].get('ai_explanation')}")
            print(f"AI Summary: {data.get('ai_summary')}")
            print("[PASSED] Search endpoint working")
            return True

        print(f"Error Response: {response.text}")
        print("[FAILED] Search endpoint returned error")
        return False

    except Exception as exc:
        print(f"[FAILED] {exc}")
        return False


def test_compare():
    """Test the /compare endpoint."""
    print_section("TEST 3: Compare Endpoint")
    try:
        params = {
            "p1_name": "Sony WH-1000XM5",
            "p1_price": 24990,
            "p1_rating": 4.7,
            "p1_platform": "Amazon",
            "p2_name": "JBL Tune 750",
            "p2_price": 12999,
            "p2_rating": 4.2,
            "p2_platform": "Flipkart",
        }
        print("Comparing:")
        print(f"  Product 1: {params['p1_name']} - Rs {params['p1_price']} - {params['p1_rating']} stars")
        print(f"  Product 2: {params['p2_name']} - Rs {params['p2_price']} - {params['p2_rating']} stars")

        response = requests.get(f"{BASE_URL}/compare", params=params, timeout=20)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Comparison: {data.get('comparison')}")
            print("[PASSED] Compare endpoint working")
            return True

        print(f"Error Response: {response.text}")
        print("[FAILED] Compare endpoint returned error")
        return False

    except Exception as exc:
        print(f"[FAILED] {exc}")
        return False


def test_filter():
    """Test the /filter endpoint."""
    print_section("TEST 4: Filter Endpoint")
    try:
        payload = {
            "products": [
                {
                    "name": "Product 1",
                    "price": 5000,
                    "rating": 4.5,
                    "review_count": 100,
                    "platform": "Amazon",
                    "url": "https://example.com/p1",
                    "image_url": "https://example.com/p1.jpg",
                },
                {
                    "name": "Product 2",
                    "price": 15000,
                    "rating": 3.5,
                    "review_count": 50,
                    "platform": "Flipkart",
                    "url": "https://example.com/p2",
                    "image_url": "https://example.com/p2.jpg",
                },
                {
                    "name": "Product 3",
                    "price": 3000,
                    "rating": 5.0,
                    "review_count": 200,
                    "platform": "Amazon",
                    "url": "https://example.com/p3",
                    "image_url": "https://example.com/p3.jpg",
                },
            ],
            "max_price": 10000,
            "min_rating": 4.0,
        }
        print("Request: Filter products with max_price=10000, min_rating=4.0")
        print(f"Input Products: {len(payload['products'])}")

        response = requests.post(f"{BASE_URL}/filter", json=payload, timeout=20)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            filtered_count = data.get("filtered_count", 0)
            print(f"Filtered Products: {filtered_count}")
            if data.get("products"):
                for product in data["products"]:
                    print(f"  - {product.get('name')}: Rs {product.get('price')} - {product.get('rating')} stars")
            print("[PASSED] Filter endpoint working")
            return True

        print(f"Error Response: {response.text}")
        print("[FAILED] Filter endpoint returned error")
        return False

    except Exception as exc:
        print(f"[FAILED] {exc}")
        return False


def main():
    """Run all tests."""
    print_section("AI SHOPPING ASSISTANT - API TESTING")
    print(f"Base URL: {BASE_URL}")

    results = []
    results.append(("Health Check", test_health()))
    results.append(("Search", test_search()))
    results.append(("Compare", test_compare()))
    results.append(("Filter", test_filter()))

    print_section("TEST SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    for test_name, result in results:
        status = "[PASSED]" if result else "[FAILED]"
        print(f"  {test_name}: {status}")

    if passed == total:
        print("\nAll tests passed! Backend is ready for frontend integration.")
    else:
        print(f"\n{total - passed} test(s) failed. Please review the errors above.")


if __name__ == "__main__":
    main()
