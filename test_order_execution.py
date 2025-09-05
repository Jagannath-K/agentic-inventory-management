"""
Test script for order execution with predicted demand and email notifications
"""

import sys
import os
import asyncio
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.planner import PlannerAgent
from agents.executor import ExecutorAgent

def test_order_execution():
    """Test the order execution with predicted demand"""
    print("🧪 Testing Order Execution with Predicted Demand")
    print("=" * 55)
    
    try:
        # Initialize agents
        print("Initializing AI agents...")
        planner = PlannerAgent()
        executor = ExecutorAgent()
        
        # Generate plans
        print("Generating inventory plans...")
        plans = planner.create_inventory_plan()
        
        if not plans:
            print("❌ No plans generated")
            return
            
        print(f"✅ Generated {len(plans)} plans")
        
        # Show first few plans
        print("\nFirst 3 plans:")
        for i, plan in enumerate(plans[:3]):
            print(f"  {i+1}. {plan.product_name}: {plan.reorder_quantity} units ({plan.urgency_level})")
        
        # Test order creation with predicted demand
        print("\n🔮 Testing order creation with predicted demand...")
        
        # Take first critical or high priority plan
        test_plan = None
        for plan in plans:
            if plan.urgency_level in ['CRITICAL', 'HIGH'] and plan.reorder_quantity > 0:
                test_plan = plan
                break
        
        if test_plan:
            print(f"Testing with: {test_plan.product_name}")
            
            # Create order request (this will use predicted demand)
            order_request = executor.create_order_request(test_plan)
            
            print(f"Original plan quantity: {test_plan.reorder_quantity}")
            print(f"AI-adjusted quantity: {order_request.quantity}")
            print(f"Estimated cost: ${order_request.estimated_cost:,.2f}")
            print(f"Supplier email: {executor.supplier_email}")
            
            # Test email sending (without actually sending)
            print("\n📧 Testing email functionality...")
            
            # Validate order
            validation = executor.validate_order_request(order_request)
            if validation['valid']:
                print("✅ Order validation passed")
                if validation['warnings']:
                    for warning in validation['warnings']:
                        print(f"⚠️  {warning}")
            else:
                print("❌ Order validation failed:")
                for issue in validation['issues']:
                    print(f"  - {issue}")
                return
            
            # Simulate order placement (this will send actual email)
            print(f"\n📤 Sending order email to {executor.supplier_email}...")
            
            async def test_order():
                result = await executor.simulate_order_placement(order_request)
                return result
                
            result = asyncio.run(test_order())
            
            if result.success:
                print(f"✅ Order {result.order_id} processed successfully!")
                print(f"   Message: {result.message}")
                print(f"   Actual cost: ${result.actual_cost:,.2f}")
            else:
                print(f"❌ Order failed: {result.message}")
                
        else:
            print("No suitable plans found for testing")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_order_execution()
