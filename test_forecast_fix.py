"""
Test the fixed forecast functionality
"""

import sys
sys.path.append('.')

from models.predictor import DemandPredictor
from datetime import datetime, timedelta
import pandas as pd

def test_forecast_variation():
    """Test if the forecast now shows variation instead of flat lines"""
    
    print("TESTING FORECAST VARIATION FIX")
    print("="*40)
    
    # Initialize predictor
    predictor = DemandPredictor()
    
    # Test with a product
    product_id = 'G001'  # Rice Basmati
    
    print(f"Testing forecast for {product_id}")
    
    # Train the model
    print("Training models...")
    training_result = predictor.train_models(product_id)
    
    if training_result:
        print(f"✅ Models trained: {list(training_result.get('models_trained', []))}")
        
        # Test predictions for different days
        print("\nTesting predictions for different days:")
        
        base_date = datetime.now()
        predictions = []
        
        for day_offset in [1, 2, 3, 7, 14, 21, 30]:
            target_date = base_date + timedelta(days=day_offset)
            
            pred_result = predictor.predict_demand(product_id, target_date)
            prediction = pred_result['predicted_demand']
            predictions.append(prediction)
            
            day_name = target_date.strftime('%A')
            print(f"Day +{day_offset:2d} ({day_name}): {prediction:.2f} units")
        
        # Check for variation
        pred_std = pd.Series(predictions).std()
        pred_mean = pd.Series(predictions).mean()
        cv = pred_std / pred_mean if pred_mean > 0 else 0
        
        print(f"\nForecast Analysis:")
        print(f"Mean prediction: {pred_mean:.2f}")
        print(f"Standard deviation: {pred_std:.2f}")
        print(f"Coefficient of variation: {cv:.3f}")
        
        if cv > 0.05:  # More than 5% variation
            print("✅ SUCCESS: Forecast shows proper variation!")
        elif cv > 0.01:  # More than 1% variation
            print("⚠️  PARTIAL: Some variation detected")
        else:
            print("❌ ISSUE: Forecast still mostly flat")
            
        # Test weekend vs weekday effect
        print("\nTesting day-of-week effects:")
        
        # Find next Monday and Saturday
        today = base_date
        days_until_monday = (7 - today.weekday()) % 7
        if days_until_monday == 0:
            days_until_monday = 7
        
        monday = today + timedelta(days=days_until_monday)
        saturday = monday + timedelta(days=5)
        
        monday_pred = predictor.predict_demand(product_id, monday)['predicted_demand']
        saturday_pred = predictor.predict_demand(product_id, saturday)['predicted_demand']
        
        print(f"Monday prediction: {monday_pred:.2f}")
        print(f"Saturday prediction: {saturday_pred:.2f}")
        
        weekend_boost = saturday_pred / monday_pred if monday_pred > 0 else 1
        print(f"Weekend boost factor: {weekend_boost:.2f}x")
        
        if weekend_boost > 1.1:
            print("✅ Weekend effect detected!")
        else:
            print("⚠️  Minimal weekend effect")
            
    else:
        print("❌ Model training failed")

if __name__ == "__main__":
    test_forecast_variation()
