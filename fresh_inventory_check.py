"""
Force Fresh Data Load and Compare All Values
"""
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def fresh_inventory_calculation():
    """Force fresh data load and calculate inventory values"""
    print("🔄 Fresh Data Load and Inventory Calculation")
    print("=" * 55)
    
    try:
        # Force fresh data load directly from CSV
        stock_data = pd.read_csv('data/stock.csv')
        
        print(f"📊 Fresh Data Loaded: {len(stock_data)} products")
        print(f"📅 Data source: data/stock.csv")
        print()
        
        # Calculate total inventory value
        total_value = (stock_data['current_stock'] * stock_data['unit_cost']).sum()
        print(f"💰 Total Inventory Value: ₹{total_value:,.2f}")
        print()
        
        # Show all individual calculations
        print("📋 Individual Product Values:")
        print("-" * 70)
        individual_sum = 0
        for _, row in stock_data.iterrows():
            value = row['current_stock'] * row['unit_cost']
            individual_sum += value
            print(f"{row['product_name']:20} | {row['current_stock']:3d} × ₹{row['unit_cost']:7.2f} = ₹{value:8.2f}")
        
        print("-" * 70)
        print(f"{'TOTAL':20} | {'':3} × {'':7} = ₹{individual_sum:8.2f}")
        print()
        
        # Verification
        if abs(total_value - individual_sum) < 0.01:
            print("✅ Calculations are consistent!")
        else:
            print(f"❌ Calculation mismatch: ₹{abs(total_value - individual_sum):.2f}")
        
        # Check for any recent updates
        print(f"\n🕒 Recent Updates Check:")
        print(f"   Most recent update: {stock_data['last_updated'].max()}")
        recent_updates = stock_data[stock_data['last_updated'].str.contains('2025-09-05', na=False)]
        print(f"   Products updated today: {len(recent_updates)}")
        
        if len(recent_updates) > 0:
            print(f"   Recently updated products:")
            for _, row in recent_updates.iterrows():
                print(f"     - {row['product_name']}: Stock={row['current_stock']}, Updated={row['last_updated']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fresh_inventory_calculation()
