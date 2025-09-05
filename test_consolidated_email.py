#!/usr/bin/env python3
"""
Test script for consolidated email functionality
"""

import asyncio
import sys
sys.path.append('.')

from agents.executor import ExecutorAgent
from agents.planner import PlannerAgent

async def test_consolidated_email():
    print("🧪 Testing Consolidated Email System")
    print("=" * 50)
    
    # Create planner and executor
    planner = PlannerAgent()
    executor = ExecutorAgent()
    
    # Generate plans
    print("📋 Generating inventory plans...")
    plans = planner.create_inventory_plan()
    print(f"Generated {len(plans)} plans")
    
    # Filter to get a few high-priority plans for testing
    high_priority_plans = [plan for plan in plans if plan.urgency_level in ['CRITICAL', 'HIGH']][:3]
    print(f"Selected {len(high_priority_plans)} high-priority plans for testing")
    
    for i, plan in enumerate(high_priority_plans, 1):
        print(f"  {i}. {plan.product_name} - {plan.urgency_level} - Reorder: {plan.reorder_quantity}")
    
    # Execute with new consolidated approach
    print("\n📧 Executing orders with consolidated email...")
    results = await executor.execute_inventory_plans(high_priority_plans)
    
    print(f"\n✅ Execution Results:")
    print(f"   • Total plans: {results['total_plans']}")
    print(f"   • Successful orders: {results['successful_orders']}")
    print(f"   • Failed orders: {results['failed_orders']}")
    print(f"   • Total cost: ₹{results['total_cost']:,.2f}")
    print(f"   • Consolidated email sent: {results.get('consolidated_email_sent', 'Not specified')}")
    print(f"   • Email message: {results.get('email_message', 'No message')}")
    
    print(f"\n📋 Order Details:")
    for order in results['order_details']:
        status = "✅" if order['success'] else "❌"
        print(f"   {status} {order['product_name']} ({order['order_id']}) - ₹{order['cost']:,.2f}")
    
    print(f"\n🎯 Summary: Instead of {len(high_priority_plans)} individual emails, sent 1 consolidated email!")

if __name__ == "__main__":
    asyncio.run(test_consolidated_email())
