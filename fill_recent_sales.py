"""
Check sales data date range and fill missing days
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def check_and_fill_recent_data():
    """Check for missing recent dates and fill them"""
    
    print("CHECKING SALES DATA COMPLETENESS")
    print("="*40)
    
    # Load current data
    df = pd.read_csv('data/sales.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Total records: {len(df)}")
    print(f"Today: {datetime.now().date()}")
    
    # Check for missing dates
    last_date = df['date'].max().date()
    today = datetime.now().date()
    missing_days = (today - last_date).days
    
    print(f"Missing days: {missing_days}")
    
    if missing_days > 0:
        print("Missing dates:")
        missing_dates = []
        current = last_date + timedelta(days=1)
        while current <= today:
            print(f"  {current}")
            missing_dates.append(current)
            current += timedelta(days=1)
        
        # Fill missing dates using the same pattern logic from fix_sales_data.py
        print(f"\n🔄 FILLING {len(missing_dates)} MISSING DAYS")
        
        # Get stock data for pricing
        stock_df = pd.read_csv('data/stock.csv')
        
        # Get recent sales patterns (last 14 days)
        recent_data = df[df['date'] >= df['date'].max() - timedelta(days=14)]
        
        # Calculate patterns by day of week for each product
        recent_data['day_of_week'] = recent_data['date'].dt.dayofweek
        patterns = recent_data.groupby(['product_id', 'product_name', 'day_of_week']).agg({
            'quantity_sold': ['mean', 'std', 'count']
        }).reset_index()
        
        patterns.columns = ['product_id', 'product_name', 'day_of_week', 'avg_qty', 'std_qty', 'count']
        patterns['std_qty'] = patterns['std_qty'].fillna(1)
        patterns['probability'] = np.minimum(patterns['count'] / patterns['count'].max(), 0.9)
        
        # Get overall patterns as fallback
        overall_patterns = recent_data.groupby(['product_id', 'product_name']).agg({
            'quantity_sold': ['mean', 'std']
        }).reset_index()
        overall_patterns.columns = ['product_id', 'product_name', 'avg_qty', 'std_qty']
        overall_patterns['std_qty'] = overall_patterns['std_qty'].fillna(1)
        
        new_records = []
        customer_counter = 60000  # Start with high number
        
        for missing_date in missing_dates:
            day_of_week = missing_date.weekday()
            
            # For each product, generate sales based on patterns
            for _, product in overall_patterns.iterrows():
                product_id = product['product_id']
                product_name = product['product_name']
                
                # Get unit cost from stock data
                unit_cost = stock_df[stock_df['product_id'] == product_id]['unit_cost'].iloc[0]
                
                # Check if we have day-specific pattern
                day_pattern = patterns[
                    (patterns['product_id'] == product_id) & 
                    (patterns['day_of_week'] == day_of_week)
                ]
                
                if not day_pattern.empty and day_pattern.iloc[0]['count'] >= 2:
                    prob_sale = day_pattern.iloc[0]['probability']
                    avg_qty = day_pattern.iloc[0]['avg_qty']
                    std_qty = day_pattern.iloc[0]['std_qty']
                else:
                    # Use overall pattern with lower probability
                    prob_sale = 0.6  # Default probability
                    avg_qty = product['avg_qty']
                    std_qty = product['std_qty']
                
                # Decide if this product has sales on this day
                if np.random.random() < prob_sale:
                    # Generate realistic sales quantity
                    qty = max(1, int(np.random.normal(avg_qty, std_qty)))
                    
                    # Limit unrealistic quantities
                    if qty > avg_qty * 2:
                        qty = int(avg_qty * 1.5)
                    
                    if qty > 0:
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
        
        if new_records:
            # Add to dataframe
            new_df = pd.DataFrame(new_records)
            combined_df = pd.concat([df, new_df], ignore_index=True)
            combined_df = combined_df.sort_values(['date', 'product_id'])
            
            # Save updated data
            combined_df.to_csv('data/sales.csv', index=False)
            
            print(f"✅ Added {len(new_records)} records for {len(missing_dates)} missing days")
            print(f"New date range: {combined_df['date'].min().date()} to {combined_df['date'].max().date()}")
            
            # Show summary by date
            new_daily = new_df.groupby('date')['quantity_sold'].sum()
            print("\nDaily sales added:")
            for date, qty in new_daily.items():
                print(f"  {date}: {qty} units")
        else:
            print("No records needed to be added")
    else:
        print("✅ Data is already up to date!")

if __name__ == "__main__":
    np.random.seed(42)  # For consistent results
    check_and_fill_recent_data()
