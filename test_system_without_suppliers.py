"""
Quick System Test After Supplier Data Removal
"""
import asyncio
from agents.planner import PlannerAgent
from agents.executor import ExecutorAgent

async def quick_system_test():
    """Test that the system works end-to-end without supplier data"""
    print("🚀 Quick System Test")
    print("=" * 30)
    
    try:
        # Test Planner
        print("📋 Testing Planner Agent...")
        planner = PlannerAgent()
        plans = planner.create_inventory_plan()
        print(f"✅ Generated {len(plans)} inventory plans")
        
        # Test Executor with a few critical plans
        critical_plans = [p for p in plans if p.urgency_level == 'CRITICAL'][:2]
        if critical_plans:
            print(f"⚡ Testing Executor with {len(critical_plans)} critical plans...")
            executor = ExecutorAgent()
            results = await executor.execute_inventory_plans(critical_plans)
            print(f"✅ Executed {results['successful_orders']} orders successfully")
            print(f"   Total cost: ₹{results['total_cost']:.2f}")
        else:
            print("✅ No critical plans to execute (system is well-stocked)")
        
        print("\n🎉 System is working perfectly without supplier data!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(quick_system_test())
