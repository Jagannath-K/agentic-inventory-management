"""
Generate realistic sales and stock data for a 3-month operational agentic system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate sales data for past 3 months (June 1 - August 26, 2025)
start_date = datetime(2025, 6, 1)
end_date = datetime(2025, 8, 26)

# Product information with Indian pricing
products = {
    'P001': {'name': 'Laptop Dell XPS', 'price': 75000, 'demand_level': 'high'},
    'P002': {'name': 'Mouse Wireless', 'price': 1200, 'demand_level': 'very_high'},
    'P003': {'name': 'Keyboard Mechanical', 'price': 3500, 'demand_level': 'high'},
    'P004': {'name': 'Monitor 24inch', 'price': 18000, 'demand_level': 'medium'},
    'P005': {'name': 'Printer HP', 'price': 12000, 'demand_level': 'low'},
    'P006': {'name': 'Webcam HD', 'price': 2800, 'demand_level': 'medium'},
    'P007': {'name': 'Headphones Bluetooth', 'price': 4500, 'demand_level': 'high'},
    'P008': {'name': 'Tablet Samsung', 'price': 25000, 'demand_level': 'medium'},
    'P009': {'name': 'Speaker Bluetooth', 'price': 2200, 'demand_level': 'medium'},
    'P010': {'name': 'External HDD', 'price': 6500, 'demand_level': 'medium'},
    'P011': {'name': 'USB Cable', 'price': 800, 'demand_level': 'very_high'},
    'P012': {'name': 'Power Bank', 'price': 2000, 'demand_level': 'high'},
    'P013': {'name': 'Phone Case', 'price': 800, 'demand_level': 'very_high'},
    'P014': {'name': 'Desk Lamp', 'price': 1500, 'demand_level': 'low'}
}

# Demand multipliers for different levels
demand_multipliers = {
    'very_high': 3.0,
    'high': 2.0,
    'medium': 1.0,
    'low': 0.4
}

# Generate sales data
sales_data = []
customer_counter = 1

current_date = start_date
while current_date <= end_date:
    # Weekend factor (lower sales on weekends)
    weekend_factor = 0.6 if current_date.weekday() >= 5 else 1.0
    
    # Monthly growth factor (business growing over time)
    month_factor = 1.0 + (current_date.month - 6) * 0.1
    
    for product_id, product_info in products.items():
        base_demand = demand_multipliers[product_info['demand_level']]
        daily_demand = base_demand * weekend_factor * month_factor
        
        # Add some randomness
        daily_sales = max(0, int(np.random.poisson(daily_demand)))
        
        if daily_sales > 0:
            # Split sales across multiple transactions if high volume
            num_transactions = min(daily_sales, random.randint(1, 3))
            remaining_qty = daily_sales
            
            for _ in range(num_transactions):
                if remaining_qty <= 0:
                    break
                    
                qty_this_transaction = min(remaining_qty, random.randint(1, max(1, remaining_qty)))
                remaining_qty -= qty_this_transaction
                
                # Price variation (seasonal/demand-based)
                price_variation = random.uniform(0.95, 1.05)
                unit_price = product_info['price'] * price_variation
                
                sales_data.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'product_id': product_id,
                    'product_name': product_info['name'],
                    'quantity_sold': qty_this_transaction,
                    'unit_price': round(unit_price, 2),
                    'customer_id': f'C{customer_counter:04d}',
                    'sales_channel': random.choice(['online', 'retail', 'wholesale']),
                    'price': ''  # Keep for compatibility
                })
                customer_counter += 1
    
    current_date += timedelta(days=1)

# Create DataFrame and save
sales_df = pd.DataFrame(sales_data)
print(f"Generated {len(sales_df)} sales records")
print(f"Date range: {sales_df['date'].min()} to {sales_df['date'].max()}")
print(f"Total revenue: ₹{(sales_df['quantity_sold'] * sales_df['unit_price']).sum():,.2f}")

# Save to CSV
sales_df.to_csv('data/sales_updated.csv', index=False)

print("\nSales data generated successfully!")
print("Sample records:")
print(sales_df.head(10))
