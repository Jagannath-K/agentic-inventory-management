"""
Fix recent sales data to match original optimized patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fix_recent_sales_patterns():
    """Remove inconsistent recent data and regenerate using original logic"""
    
    print("FIXING RECENT SALES PATTERN INCONSISTENCY")
    print("="*45)
    
    # Load current data
    df = pd.read_csv('data/sales.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Keep only data up to Sep 5 (the original optimized data)
    original_data = df[df['date'] <= '2025-09-05'].copy()
    
    print(f"Original data: {len(df)} records")
    print(f"Keeping data up to 2025-09-05: {len(original_data)} records")
    print(f"Removing inconsistent recent data: {len(df) - len(original_data)} records")
    
    # Get stock data for pricing
    stock_df = pd.read_csv('data/stock.csv')
    
    # Dates to regenerate (Sep 6 to today)
    start_date = datetime(2025, 9, 6).date()
    end_date = datetime.now().date()
    
    dates_to_fill = []
    current_date = start_date
    while current_date <= end_date:
        dates_to_fill.append(current_date)
        current_date += timedelta(days=1)
    
    print(f"Regenerating {len(dates_to_fill)} days: {start_date} to {end_date}")
    
    # Use the EXACT same logic from fix_sales_data.py for pattern generation
    new_records = []
    customer_counter = 70000  # Start with high number
    
    for missing_date in dates_to_fill:
        # For each product, generate sales using the SAME category-based logic
        for _, stock_product in stock_df.iterrows():
            product_id = stock_product['product_id']
            product_name = stock_product['product_name']
            category = stock_product['category']
            unit_cost = stock_product['unit_cost']
            
            # EXACT SAME LOGIC as fix_sales_data.py
            # Set base demand levels based on category
            if category in ['Staples', 'Dairy']:
                base_demand = np.random.choice([8, 10, 12], p=[0.3, 0.4, 0.3])  # High demand
                weekend_factor = 1.4  # Higher weekend demand
            elif category in ['Vegetables', 'Bakery']:
                base_demand = np.random.choice([5, 6, 7], p=[0.3, 0.4, 0.3])   # Medium demand
                weekend_factor = 1.2
            elif category in ['Spices', 'Condiments']:
                base_demand = np.random.choice([2, 3, 4], p=[0.3, 0.4, 0.3])   # Low but steady
                weekend_factor = 1.1
            else:
                base_demand = np.random.choice([4, 5, 6], p=[0.3, 0.4, 0.3])   # Medium
                weekend_factor = 1.3
            
            # Day of week patterns (0=Monday, 6=Sunday)
            day_of_week = missing_date.weekday()
            
            # Weekly pattern
            if day_of_week in [5, 6]:  # Weekend
                daily_demand = int(base_demand * weekend_factor)
            elif day_of_week in [0, 1]:  # Monday, Tuesday (stock up days)
                daily_demand = int(base_demand * 1.2)
            else:  # Mid-week
                daily_demand = base_demand
            
            # Add monthly seasonality (very subtle)
            month = missing_date.month
            if month in [6, 7, 8]:  # Summer months
                if category == 'Beverages':
                    daily_demand = int(daily_demand * 1.3)
                elif category == 'Dairy':
                    daily_demand = int(daily_demand * 0.9)  # Less milk in summer
            
            # Add small random variation (±20% max)
            variation = np.random.uniform(0.85, 1.15)
            daily_demand = max(1, int(daily_demand * variation))
            
            # Create realistic transaction breakdown
            if daily_demand > 0:
                # Split into 1-4 transactions with realistic quantities
                if daily_demand <= 3:
                    transactions = [daily_demand]
                elif daily_demand <= 8:
                    # Split into 2-3 transactions
                    num_trans = np.random.choice([2, 3], p=[0.7, 0.3])
                    transactions = []
                    remaining = daily_demand
                    for i in range(num_trans - 1):
                        qty = np.random.randint(1, max(2, remaining - (num_trans - i - 1)))
                        transactions.append(qty)
                        remaining -= qty
                    transactions.append(remaining)
                else:
                    # Split into 3-4 transactions
                    num_trans = np.random.choice([3, 4], p=[0.6, 0.4])
                    transactions = []
                    remaining = daily_demand
                    for i in range(num_trans - 1):
                        qty = np.random.randint(1, max(2, remaining - (num_trans - i - 1)))
                        transactions.append(qty)
                        remaining -= qty
                    transactions.append(remaining)
                
                # Create records for each transaction
                for i, qty in enumerate(transactions):
                    new_records.append({
                        'date': missing_date.strftime('%Y-%m-%d'),
                        'product_id': product_id,
                        'product_name': product_name,
                        'quantity_sold': qty,
                        'unit_price': unit_cost,  # Use consistent pricing
                        'customer_id': f'C{customer_counter:04d}',
                        'price': '',
                        'sales_channel': ''
                    })
                    customer_counter += 1
    
    # Create new complete dataset
    new_df = pd.DataFrame(new_records)
    
    # Convert original data date back to string format for consistency
    original_data['date'] = original_data['date'].dt.strftime('%Y-%m-%d')
    
    complete_df = pd.concat([original_data, new_df], ignore_index=True)
    complete_df = complete_df.sort_values(['date', 'product_id'])
    
    # Save the corrected dataset
    complete_df.to_csv('data/sales.csv', index=False)
    
    print(f"✅ Generated {len(new_records)} new records using original logic")
    print(f"✅ Total records: {len(complete_df)}")
    
    # Verify pattern consistency
    print("\\n📊 PATTERN VERIFICATION")
    complete_df['date'] = pd.to_datetime(complete_df['date'])
    
    before_sep5 = complete_df[complete_df['date'] <= '2025-09-05']
    after_sep5 = complete_df[complete_df['date'] > '2025-09-05']
    
    before_daily = before_sep5.groupby('date')['quantity_sold'].sum()
    after_daily = after_sep5.groupby('date')['quantity_sold'].sum()
    
    print(f"Before Sep 5: Avg daily = {before_daily.mean():.1f} units")
    print(f"After Sep 5: Avg daily = {after_daily.mean():.1f} units")
    print(f"Pattern consistency: {abs(before_daily.mean() - after_daily.mean()) / before_daily.mean() * 100:.1f}% difference")
    
    if abs(before_daily.mean() - after_daily.mean()) / before_daily.mean() < 0.15:  # <15% difference
        print("✅ Patterns are now consistent!")
    else:
        print("⚠️ Some pattern difference remains")
    
    # Show daily sales for verification
    print("\\nNew daily sales pattern:")
    for date, qty in after_daily.items():
        print(f"  {date.strftime('%Y-%m-%d')}: {qty} units")

if __name__ == "__main__":
    np.random.seed(42)  # Same seed for consistency
    fix_recent_sales_patterns()
