"""
Debug Cost Differences Between Planning and Execution
"""
import asyncio
from agents.planner import PlannerAgent
from agents.executor import ExecutorAgent

async def debug_cost_differences():
    """Compare planning vs execution costs in detail"""
    print("🔍 Debugging Cost Differences")
    print("=" * 50)
    
    # Generate plans
    planner = PlannerAgent()
    plans = planner.create_inventory_plan()
    
    # Filter for actionable plans (like the UI does)
    actionable_plans = [p for p in plans if p.urgency_level in ['CRITICAL', 'HIGH', 'MEDIUM'] and p.reorder_quantity > 0]
    
    print(f"📋 Planning Details:")
    print(f"Total actionable plans: {len(actionable_plans)}")
    
    planning_total = 0
    plan_details = []
    
    for plan in actionable_plans:
        unit_cost = planner.stock_data[planner.stock_data['product_id'] == plan.product_id]['unit_cost'].iloc[0]
        cost = plan.reorder_quantity * unit_cost
        planning_total += cost
        plan_details.append({
            'product': plan.product_name,
            'quantity': plan.reorder_quantity,
            'unit_cost': unit_cost,
            'total_cost': cost
        })
        print(f"  {plan.product_name}: {plan.reorder_quantity} × ₹{unit_cost:.2f} = ₹{cost:.2f}")
    
    print(f"📊 Planning Total: ₹{planning_total:.2f}")
    print()
    
    # Execute orders
    executor = ExecutorAgent()
    results = await executor.execute_inventory_plans(actionable_plans)
    
    print(f"⚡ Execution Details:")
    print(f"Successful orders: {results['successful_orders']}")
    print(f"Failed orders: {results['failed_orders']}")
    print(f"Total cost: ₹{results['total_cost']:.2f}")
    
    print()
    print(f"💰 Cost Comparison:")
    print(f"Planning: ₹{planning_total:.2f}")
    print(f"Execution: ₹{results['total_cost']:.2f}")
    print(f"Difference: ₹{results['total_cost'] - planning_total:.2f}")
    print(f"Percentage: {abs(results['total_cost'] - planning_total) / planning_total * 100:.2f}%")

if __name__ == "__main__":
    asyncio.run(debug_cost_differences())
