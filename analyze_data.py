import pandas as pd

# Load and analyze sales data
sales = pd.read_csv('data/sales.csv')
stock = pd.read_csv('data/stock.csv')

print("=== AGENTIC INVENTORY SYSTEM - DATA ANALYSIS ===")
print(f"Sales Records: {len(sales)}")
print(f"Date Range: {sales['date'].min()} to {sales['date'].max()}")
print(f"Unique Products: {sales['product_id'].nunique()}")
print(f"Total Revenue: ₹{sales['unit_price'].sum():,.2f}")
print(f"Stock Items: {len(stock)}")
print(f"Total Stock Value: ₹{(stock['current_stock'] * stock['unit_cost']).sum():,.2f}")
