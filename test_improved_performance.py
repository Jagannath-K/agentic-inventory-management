"""
Test ML model performance with improved sales data
"""

import pandas as pd
import numpy as np
from models.predictor import DemandPredictor
from datetime import datetime, timedelta

def test_improved_performance():
    """Test prediction accuracy with improved data"""
    
    print("TESTING ML MODEL PERFORMANCE WITH IMPROVED DATA")
    print("="*50)
    
    # Initialize predictor
    predictor = DemandPredictor()
    
    # Test with multiple products
    test_products = ['G001', 'G002', 'G003', 'G014', 'G015']
    
    all_errors = []
    
    for product_id in test_products:
        print(f"\n🧪 Testing {product_id}")
        
        try:
            # Train model
            result = predictor.train_models(product_id)
            
            if result['success']:
                # Get model performance
                performance = result.get('model_performance', {})
                if performance:
                    best_model = min(performance.keys(), key=lambda k: performance[k]['test_mae'])
                    test_mae = performance[best_model]['test_mae']
                    
                    # Calculate percentage error
                    sales_data = pd.read_csv('data/sales.csv')
                    product_sales = sales_data[sales_data['product_id'] == product_id]
                    daily_sales = product_sales.groupby('date')['quantity_sold'].sum()
                    avg_daily_sales = daily_sales.mean()
                    
                    percentage_error = (test_mae / avg_daily_sales) * 100
                    all_errors.append(percentage_error)
                    
                    print(f"  Best model: {best_model}")
                    print(f"  MAE: {test_mae:.2f}")
                    print(f"  Avg daily sales: {avg_daily_sales:.2f}")
                    print(f"  Error percentage: {percentage_error:.1f}%")
                else:
                    print(f"  ❌ No performance data available")
            else:
                print(f"  ❌ Training failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"  ❌ Error testing {product_id}: {e}")
    
    if all_errors:
        avg_error = np.mean(all_errors)
        print(f"\n📊 OVERALL PERFORMANCE")
        print(f"Average prediction error: {avg_error:.1f}%")
        print(f"Target error: <25%")
        
        if avg_error < 25:
            print("✅ SUCCESS: Error reduced to target level!")
        elif avg_error < 35:
            print("🟡 GOOD: Significant improvement achieved")
        else:
            print("🔴 NEEDS MORE WORK: Error still high")
            
        print(f"\nError by product:")
        for i, product_id in enumerate(test_products[:len(all_errors)]):
            print(f"  {product_id}: {all_errors[i]:.1f}%")

if __name__ == "__main__":
    test_improved_performance()
