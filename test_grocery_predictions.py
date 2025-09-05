#!/usr/bin/env python3
"""
Test Grocery Retail AI Prediction with Reasoning
"""

from models.predictor import DemandPredictor
from datetime import datetime, timedelta

def test_grocery_predictions():
    print("🏪 Testing AI-Powered Grocery Retail Predictions")
    print("=" * 60)
    
    dp = DemandPredictor()
    dp.load_data()
    
    # Test different grocery products
    test_products = [
        ('G001', 'Rice Basmati 1kg'),
        ('G006', 'Milk Packet 500ml'), 
        ('G011', 'Biscuits Parle-G'),
        ('G015', 'Onions 1kg')
    ]
    
    # Test different scenarios
    test_dates = [
        (datetime(2025, 9, 1), "Month Beginning (Salary Day)"),
        (datetime(2025, 9, 7), "Weekend Shopping"),
        (datetime(2025, 9, 15), "Mid-Month Period"),
        (datetime(2025, 9, 28), "Month End Period")
    ]
    
    for product_id, product_name in test_products:
        print(f"\n📦 Product: {product_name} ({product_id})")
        print("-" * 40)
        
        for test_date, scenario in test_dates:
            try:
                result = dp.predict_demand(product_id, test_date)
                
                print(f"\n🗓️  {scenario} - {test_date.strftime('%Y-%m-%d')}")
                print(f"   Predicted Demand: {result['predicted_demand']:.1f} units")
                print(f"   Confidence: {result['confidence']:.1%}")
                print(f"   Method: {result['method']}")
                
                # Display reasoning if available
                if 'reasoning' in result and result['reasoning']:
                    print(f"   🤖 AI Reasoning:")
                    reasoning_parts = result['reasoning'].split(' | ')
                    for part in reasoning_parts[:3]:  # Show first 3 reasoning points
                        print(f"      • {part}")
                        
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    print(f"\n🎯 Grocery retail AI prediction testing complete!")

if __name__ == "__main__":
    test_grocery_predictions()
