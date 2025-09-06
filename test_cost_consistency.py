#!/usr/bin/env python3
"""
Cost Consistency Verification Test
"""

import asyncio
import sys
sys.path.append('.')

from agents.executor import ExecutorAgent
from agents.planner import PlannerAgent

async def verify_cost_consistency():
    print("💰 Cost Consistency Verification Test")
    print("=" * 50)
    
    # Create planner and executor
    planner = PlannerAgent()
    executor = ExecutorAgent()
    
    # Generate plans
    print("📋 Generating inventory plans...")
    plans = planner.create_inventory_plan()
    
    # Filter to get high-priority plans
    high_priority_plans = [plan for plan in plans if plan.urgency_level in ['CRITICAL', 'HIGH']][:3]
    print(f"Selected {len(high_priority_plans)} high-priority plans for cost verification")
    
    # Calculate estimated reorder investment from planning
    # We need to get unit costs from stock data
    import pandas as pd
    stock_data = pd.read_csv('data/stock.csv')
    stock_dict = stock_data.set_index('product_id')['unit_cost'].to_dict()
    
    planning_total = sum(plan.reorder_quantity * stock_dict.get(plan.product_id, 0) for plan in high_priority_plans)
    print(f"📊 Planning Agent Reorder Investment: ₹{planning_total:,.2f}")
    
    # Execute orders and get actual costs
    print("\n📧 Executing orders...")
    results = await executor.execute_inventory_plans(high_priority_plans)
    
    execution_total = results['total_cost']
    print(f"⚡ Execution Agent Total Cost: ₹{execution_total:,.2f}")
    
    # Verify individual order costs sum correctly
    individual_total = sum(order['cost'] for order in results['order_details'])
    print(f"🧮 Sum of Individual Orders: ₹{individual_total:,.2f}")
    
    # Check consistency
    print(f"\n✅ Cost Consistency Check:")
    print(f"   Planning vs Execution: ₹{planning_total:,.2f} vs ₹{execution_total:,.2f}")
    print(f"   Difference: ₹{abs(planning_total - execution_total):,.2f}")
    print(f"   Execution vs Individual Sum: ₹{execution_total:,.2f} vs ₹{individual_total:,.2f}")
    print(f"   Difference: ₹{abs(execution_total - individual_total):,.2f}")
    
    # Check if differences are within acceptable range (cost variations)
    planning_diff_pct = abs(planning_total - execution_total) / planning_total * 100
    execution_diff_pct = abs(execution_total - individual_total) / execution_total * 100
    
    print(f"\n📈 Variance Analysis:")
    print(f"   Planning vs Execution: {planning_diff_pct:.2f}% (Expected: 0.5-1.0% due to supplier pricing)")
    print(f"   Execution vs Individual: {execution_diff_pct:.2f}% (Expected: 0.0%)")
    
    if execution_diff_pct < 0.01:  # Less than 0.01%
        print(f"   ✅ EXCELLENT: Execution totals are perfectly consistent!")
    else:
        print(f"   ❌ WARNING: Execution totals don't match!")
    
    if planning_diff_pct < 2.0:  # Less than 2%
        print(f"   ✅ GOOD: Planning vs Execution within acceptable range!")
    else:
        print(f"   ❌ WARNING: Planning vs Execution difference too large!")

if __name__ == "__main__":
    asyncio.run(verify_cost_consistency())
