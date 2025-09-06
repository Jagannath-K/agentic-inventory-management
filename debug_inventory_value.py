"""
Debug Inventory Value Calculation Discrepancy
"""
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ui.app import load_data

def debug_inventory_value_discrepancy():
    """Debug the discrepancy between overview and stock status inventory values"""
    print("🔍 Debugging Inventory Value Discrepancy")
    print("=" * 50)
    
    try:
        # Load data
        sales_data, stock_data = load_data()
        
        if stock_data is None or len(stock_data) == 0:
            print("❌ No stock data available")
            return
        
        print(f"📊 Loaded {len(stock_data)} products")
        print()
        
        # Calculate overview inventory value (same as line 203)
        overview_total = (stock_data['current_stock'] * stock_data['unit_cost']).sum()
        print(f"💰 Overview Calculation:")
        print(f"   Total: ₹{overview_total:,.2f}")
        print()
        
        # Calculate individual values (same as stock status table)
        print(f"📋 Individual Product Values:")
        individual_total = 0
        for _, row in stock_data.iterrows():
            product_value = row['current_stock'] * row['unit_cost']
            individual_total += product_value
            print(f"   {row['product_name']}: {row['current_stock']} × ₹{row['unit_cost']:.2f} = ₹{product_value:.2f}")
        
        print()
        print(f"📊 Sum of Individual Values: ₹{individual_total:,.2f}")
        print()
        
        # Compare
        difference = abs(overview_total - individual_total)
        print(f"🔍 Comparison:")
        print(f"   Overview: ₹{overview_total:,.2f}")
        print(f"   Individual Sum: ₹{individual_total:,.2f}")
        print(f"   Difference: ₹{difference:.2f}")
        
        if difference < 0.01:
            print("✅ Values match - issue might be in data display or caching")
        else:
            print("❌ Values don't match - calculation error found!")
            
        # Check for any data inconsistencies
        print(f"\n🔍 Data Quality Check:")
        print(f"   Null current_stock: {stock_data['current_stock'].isnull().sum()}")
        print(f"   Null unit_cost: {stock_data['unit_cost'].isnull().sum()}")
        print(f"   Zero current_stock: {(stock_data['current_stock'] == 0).sum()}")
        print(f"   Zero unit_cost: {(stock_data['unit_cost'] == 0).sum()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_inventory_value_discrepancy()
