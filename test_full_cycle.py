#!/usr/bin/env python3
"""
Direct test of the full inventory cycle with consolidated emails
"""

import asyncio
import sys
sys.path.append('.')

from main import InventoryManagementSystem

async def test_full_cycle():
    print("🔄 Testing Full Inventory Cycle with Consolidated Emails")
    print("=" * 60)
    
    # Create system
    system = InventoryManagementSystem()
    
    print("📊 Running complete inventory cycle...")
    
    # Run one complete cycle
    results = await system.run_inventory_cycle()
    
    print(f"\n✅ Cycle Results:")
    print(f"   • Plans generated: {results.get('plans_generated', 0)}")
    print(f"   • Orders executed: {results.get('orders_executed', 0)}")
    print(f"   • Total cost: ₹{results.get('total_cost', 0):,.2f}")
    print(f"   • Insights generated: {results.get('insights_generated', 0)}")
    
    # Check if consolidated email was sent
    if 'email_message' in results:
        print(f"   📧 Email status: {results['email_message']}")
    
    print(f"\n📧 Email Consolidation Summary:")
    if results.get('orders_executed', 0) > 1:
        print(f"   ✅ SUCCESS: {results['orders_executed']} orders sent in 1 consolidated email!")
        print(f"   📉 Reduced email volume by {results['orders_executed'] - 1} emails")
    elif results.get('orders_executed', 0) == 1:
        print(f"   ✅ 1 order sent in 1 email (optimal)")
    else:
        print(f"   ℹ️ No orders executed in this cycle")
    
    print(f"\n📋 Check your email (jagannath.backup.2005@gmail.com) for the consolidated order!")

if __name__ == "__main__":
    asyncio.run(test_full_cycle())
