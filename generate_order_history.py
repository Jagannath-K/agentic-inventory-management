"""
Generate realistic order history for the past 3 months
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed
np.random.seed(42)
random.seed(42)

order_history = []
order_counter = 800  # Start from 800 to show system has been running

# Generate orders for past 3 months
start_date = datetime(2025, 6, 1)
end_date = datetime(2025, 8, 25)

products = [
    'P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 
    'P008', 'P009', 'P010', 'P011', 'P012', 'P013', 'P014'
]

product_names = {
    'P001': 'Laptop Dell XPS',
    'P002': 'Mouse Wireless', 
    'P003': 'Keyboard Mechanical',
    'P004': 'Monitor 24inch',
    'P005': 'Printer HP',
    'P006': 'Webcam HD',
    'P007': 'Headphones Bluetooth',
    'P008': 'Tablet Samsung',
    'P009': 'Speaker Bluetooth',
    'P010': 'External HDD',
    'P011': 'USB Cable',
    'P012': 'Power Bank',
    'P013': 'Phone Case',
    'P014': 'Desk Lamp'
}

unit_costs = {
    'P001': 62500, 'P002': 1000, 'P003': 2800, 'P004': 15000,
    'P005': 10000, 'P006': 2300, 'P007': 3800, 'P008': 20000,
    'P009': 1800, 'P010': 5500, 'P011': 650, 'P012': 1600,
    'P013': 650, 'P014': 1200
}

current_date = start_date
while current_date <= end_date:
    # Generate 1-3 orders per week
    if random.random() < 0.4:  # 40% chance of order on any given day
        num_orders = random.randint(1, 3)
        
        for _ in range(num_orders):
            product_id = random.choice(products)
            
            # Higher quantities for high-demand items
            if product_id in ['P001', 'P008']:  # Laptops, tablets
                quantity = random.randint(5, 25)
            elif product_id in ['P002', 'P011', 'P013']:  # High volume items
                quantity = random.randint(20, 100)
            else:
                quantity = random.randint(10, 50)
            
            total_cost = quantity * unit_costs[product_id]
            
            order_history.append({
                'order_id': f'ORD-{order_counter:06d}',
                'date': current_date.strftime('%Y-%m-%d'),
                'product_id': product_id,
                'product_name': product_names[product_id],
                'quantity_ordered': quantity,
                'unit_cost': unit_costs[product_id],
                'total_cost': total_cost,
                'supplier_email': 'jagannath.backup.2005@gmail.com',
                'status': random.choice(['Delivered', 'Delivered', 'Delivered', 'In Transit']),
                'ai_predicted_demand': random.randint(int(quantity * 0.8), int(quantity * 1.2)),
                'actual_demand_30days': random.randint(int(quantity * 0.7), int(quantity * 1.3))
            })
            
            order_counter += 1
    
    current_date += timedelta(days=1)

# Create DataFrame
order_df = pd.DataFrame(order_history)
print(f"Generated {len(order_df)} order records")
print(f"Total orders value: ₹{order_df['total_cost'].sum():,.2f}")
print(f"Orders per month: {len(order_df) / 3:.1f}")

# Save to CSV
order_df.to_csv('data/order_history.csv', index=False)
print("\nOrder history saved to data/order_history.csv")
print("\nSample orders:")
print(order_df.head(10))
