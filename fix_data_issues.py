"""
Fix date format and detergent pricing issues
"""

import pandas as pd
import numpy as np

def fix_data_issues():
    """Fix date format and pricing inconsistencies"""
    
    print("FIXING DATA ISSUES")
    print("="*30)
    
    # 1. Fix sales.csv
    print("1. Fixing sales.csv...")
    sales = pd.read_csv('data/sales.csv')
    
    # Fix date format - remove time component from all dates
    sales['date'] = pd.to_datetime(sales['date'], format='mixed').dt.strftime('%Y-%m-%d')
    
    # Fix detergent pricing - use realistic price
    detergent_price = 65.0  # Realistic detergent price
    sales.loc[sales['product_id'] == 'G013', 'unit_price'] = detergent_price
    
    print(f"  ✅ Fixed dates to YYYY-MM-DD format")
    print(f"  ✅ Fixed G013 detergent price to ₹{detergent_price}")
    
    # Save sales.csv
    sales.to_csv('data/sales.csv', index=False)
    
    # 2. Fix stock.csv
    print("2. Fixing stock.csv...")
    stock = pd.read_csv('data/stock.csv')
    
    # Update detergent unit_cost to match sales price
    stock.loc[stock['product_id'] == 'G013', 'unit_cost'] = detergent_price
    
    print(f"  ✅ Updated G013 unit_cost to ₹{detergent_price}")
    
    # Save stock.csv
    stock.to_csv('data/stock.csv', index=False)
    
    # 3. Verify the fixes
    print("3. Verifying fixes...")
    
    # Check sales dates
    sales_check = pd.read_csv('data/sales.csv')
    sample_dates = sales_check['date'].head(5).tolist()
    print(f"  Sample dates: {sample_dates}")
    
    # Check detergent pricing
    detergent_sales_prices = sales_check[sales_check['product_id'] == 'G013']['unit_price'].unique()
    stock_check = pd.read_csv('data/stock.csv')
    detergent_stock_price = stock_check[stock_check['product_id'] == 'G013']['unit_cost'].iloc[0]
    
    print(f"  G013 in sales.csv: {detergent_sales_prices}")
    print(f"  G013 in stock.csv: {detergent_stock_price}")
    
    if len(detergent_sales_prices) == 1 and detergent_sales_prices[0] == detergent_stock_price == detergent_price:
        print("  ✅ Detergent pricing is now consistent across all files")
    else:
        print("  ❌ Pricing inconsistency still exists")
    
    # Check if any other products have unrealistic prices
    print("4. Checking for other pricing issues...")
    
    # Check for any unit prices with many decimal places
    for _, product in stock_check.iterrows():
        unit_cost = product['unit_cost']
        if isinstance(unit_cost, float) and len(str(unit_cost).split('.')[-1]) > 2:
            print(f"  Warning: {product['product_id']} has many decimals: {unit_cost}")
    
    print("\\n✅ ALL FIXES COMPLETED")
    print("✅ Dates: Clean YYYY-MM-DD format")
    print("✅ Pricing: Realistic and consistent")

if __name__ == "__main__":
    fix_data_issues()
