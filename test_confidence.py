"""
Test the improved confidence scores with realistic data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.planner import PlannerAgent

def test_confidence_scores():
    print("🔍 Testing Improved Confidence Scores")
    print("=" * 40)
    
    planner = PlannerAgent()
    plans = planner.create_inventory_plan()
    
    print(f"Generated {len(plans)} plans")
    print("\nConfidence Score Analysis:")
    print("-" * 50)
    
    confidence_scores = [plan.confidence_score for plan in plans]
    avg_confidence = sum(confidence_scores) / len(confidence_scores)
    
    print(f"Average Confidence: {avg_confidence:.1%}")
    print(f"Highest Confidence: {max(confidence_scores):.1%}")
    print(f"Lowest Confidence: {min(confidence_scores):.1%}")
    
    print("\nTop 5 Most Confident Predictions:")
    sorted_plans = sorted(plans, key=lambda x: x.confidence_score, reverse=True)
    
    for i, plan in enumerate(sorted_plans[:5]):
        print(f"{i+1}. {plan.product_name}: {plan.confidence_score:.1%}")
        print(f"   Current Stock: {plan.current_stock}, Predicted Demand: {plan.predicted_demand:.1f}")
        print(f"   Urgency: {plan.urgency_level}")
        print()

if __name__ == "__main__":
    test_confidence_scores()
