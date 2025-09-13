"""
DETAILED ANALYSIS: Training Data vs Random Variations
Analyze exactly how the forecast variations are generated
"""

import sys
sys.path.append('.')

from models.predictor import DemandPredictor
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def analyze_prediction_source():
    """Analyze the source of forecast variations - training data or random"""
    
    print("DETAILED ANALYSIS: FORECAST VARIATION SOURCE")
    print("="*60)
    
    # Initialize predictor
    predictor = DemandPredictor()
    
    # Test with G001 (Rice Basmati)
    product_id = 'G001'
    
    print(f"Analyzing predictions for {product_id}")
    print("-" * 40)
    
    # Train the model first
    training_result = predictor.train_models(product_id)
    
    # Test date
    test_date_monday = datetime(2025, 9, 15)  # Monday
    test_date_saturday = datetime(2025, 9, 20)  # Saturday
    
    print("\\n1. FEATURE ANALYSIS:")
    print("=" * 25)
    
    # Get features for both dates
    features_monday = predictor.create_prediction_features(product_id, test_date_monday)
    features_saturday = predictor.create_prediction_features(product_id, test_date_saturday)
    
    print("Features for Monday vs Saturday:")
    feature_cols = features_monday.columns
    
    for col in feature_cols:
        mon_val = features_monday[col].iloc[0]
        sat_val = features_saturday[col].iloc[0]
        if mon_val != sat_val:
            print(f"  {col:15}: Monday={mon_val:8.3f}, Saturday={sat_val:8.3f}")
    
    print("\\n2. MODEL PREDICTION BREAKDOWN:")
    print("=" * 35)
    
    # Get detailed predictions
    pred_monday = predictor.predict_demand(product_id, test_date_monday)
    pred_saturday = predictor.predict_demand(product_id, test_date_saturday)
    
    print("Monday predictions by model:")
    for model_name, pred in pred_monday['model_predictions'].items():
        print(f"  {model_name:20}: {pred:6.2f}")
    print(f"  {'Ensemble Average':20}: {pred_monday['ensemble_prediction']:6.2f}")
    
    print("\\nSaturday predictions by model:")
    for model_name, pred in pred_saturday['model_predictions'].items():
        print(f"  {model_name:20}: {pred:6.2f}")
    print(f"  {'Ensemble Average':20}: {pred_saturday['ensemble_prediction']:6.2f}")
    
    print("\\n3. TRAINING DATA ANALYSIS:")
    print("=" * 30)
    
    # Load sales data to analyze training patterns
    sales_df = pd.read_csv('data/sales.csv')
    sales_df['date'] = pd.to_datetime(sales_df['date'])
    
    # Get product sales
    product_sales = sales_df[sales_df['product_id'] == product_id]
    daily_sales = product_sales.groupby('date')['quantity_sold'].sum()
    
    # Add day of week
    daily_sales_df = daily_sales.reset_index()
    daily_sales_df['day_of_week'] = daily_sales_df['date'].dt.day_name()
    
    # Calculate average by day of week
    weekday_avg = daily_sales_df.groupby('day_of_week')['quantity_sold'].agg(['mean', 'std', 'count'])
    
    print("Historical sales patterns by day of week:")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    for day in day_order:
        if day in weekday_avg.index:
            avg = weekday_avg.loc[day, 'mean']
            std = weekday_avg.loc[day, 'std']
            count = weekday_avg.loc[day, 'count']
            print(f"  {day:10}: Avg={avg:5.2f}, Std={std:5.2f}, Count={count:3.0f}")
    
    print("\\n4. PREDICTION CONSISTENCY TEST:")
    print("=" * 35)
    
    # Test same date multiple times
    print("Testing same date (Monday) 5 times:")
    monday_predictions = []
    for i in range(5):
        pred = predictor.predict_demand(product_id, test_date_monday)
        monday_predictions.append(pred['ensemble_prediction'])
        print(f"  Run {i+1}: {pred['ensemble_prediction']:6.2f}")
    
    pred_std = np.std(monday_predictions)
    print(f"\\nStandard deviation across runs: {pred_std:.6f}")
    
    if pred_std < 0.001:
        print("✅ DETERMINISTIC: Predictions are consistent (training-based)")
    else:
        print("❌ RANDOM: Predictions vary between runs")
    
    print("\\n5. CONCLUSION:")
    print("=" * 15)
    
    weekend_boost = pred_saturday['ensemble_prediction'] / pred_monday['ensemble_prediction']
    
    if pred_std < 0.001:  # Deterministic
        if weekend_boost > 1.1:  # Significant weekend effect
            print("✅ TRAINING DATA-BASED VARIATIONS:")
            print("   • Predictions are deterministic (same input = same output)")
            print("   • Variations come from learned patterns in training data")
            print("   • Weekend boost reflects actual historical sales patterns")
            print(f"   • Weekend boost factor: {weekend_boost:.2f}x")
        else:
            print("⚠️  LIMITED PATTERN LEARNING:")
            print("   • Predictions are deterministic but show minimal day effects")
    else:
        print("❌ RANDOM VARIATIONS:")
        print("   • Predictions vary between runs (indicates randomness)")
        print("   • Not based on learned patterns")
    
    print("\\n6. VERIFICATION:")
    print("=" * 15)
    
    # Check if the weekend effect matches historical data
    if 'Monday' in weekday_avg.index and 'Saturday' in weekday_avg.index:
        historical_monday = weekday_avg.loc['Monday', 'mean']
        historical_saturday = weekday_avg.loc['Saturday', 'mean']
        historical_boost = historical_saturday / historical_monday
        
        predicted_monday = pred_monday['ensemble_prediction']
        predicted_saturday = pred_saturday['ensemble_prediction']
        predicted_boost = predicted_saturday / predicted_monday
        
        print(f"Historical weekend boost: {historical_boost:.2f}x")
        print(f"Predicted weekend boost:  {predicted_boost:.2f}x")
        print(f"Difference: {abs(predicted_boost - historical_boost):.2f}x")
        
        if abs(predicted_boost - historical_boost) < 0.5:
            print("✅ Model learned historical patterns!")
        else:
            print("⚠️  Model patterns differ from historical data")

if __name__ == "__main__":
    analyze_prediction_source()
