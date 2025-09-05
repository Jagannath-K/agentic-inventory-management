#!/usr/bin/env python3
"""
Grocery Retail Data Generator for Micro Enterprise AI System
Generates realistic Indian grocery store sales patterns with intelligent spikes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import csv

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_grocery_products():
    """Generate realistic grocery products for Indian micro enterprises"""
    products = [
        # Staples (High volume, low margin)
        {'id': 'G001', 'name': 'Rice Basmati 1kg', 'category': 'Staples', 'cost': 85, 'margin': 0.15},
        {'id': 'G002', 'name': 'Wheat Flour 1kg', 'category': 'Staples', 'cost': 35, 'margin': 0.12},
        {'id': 'G003', 'name': 'Sugar 1kg', 'category': 'Staples', 'cost': 45, 'margin': 0.10},
        {'id': 'G004', 'name': 'Dal Toor 500g', 'category': 'Pulses', 'cost': 120, 'margin': 0.18},
        {'id': 'G005', 'name': 'Oil Sunflower 1L', 'category': 'Cooking Oil', 'cost': 140, 'margin': 0.14},
        
        # Daily Essentials (Medium volume, medium margin)
        {'id': 'G006', 'name': 'Milk Packet 500ml', 'category': 'Dairy', 'cost': 24, 'margin': 0.08},
        {'id': 'G007', 'name': 'Bread White 400g', 'category': 'Bakery', 'cost': 28, 'margin': 0.20},
        {'id': 'G008', 'name': 'Eggs 12 pieces', 'category': 'Protein', 'cost': 75, 'margin': 0.15},
        {'id': 'G009', 'name': 'Tea Powder 250g', 'category': 'Beverages', 'cost': 180, 'margin': 0.22},
        {'id': 'G010', 'name': 'Salt 1kg', 'category': 'Spices', 'cost': 18, 'margin': 0.25},
        
        # Packaged Goods (Lower volume, higher margin)
        {'id': 'G011', 'name': 'Biscuits Parle-G', 'category': 'Snacks', 'cost': 12, 'margin': 0.30},
        {'id': 'G012', 'name': 'Soap Lifebouy', 'category': 'Personal Care', 'cost': 35, 'margin': 0.25},
        {'id': 'G013', 'name': 'Detergent 1kg', 'category': 'Household', 'cost': 180, 'margin': 0.20},
        {'id': 'G014', 'name': 'Turmeric Powder 100g', 'category': 'Spices', 'cost': 45, 'margin': 0.28},
        {'id': 'G015', 'name': 'Onions 1kg', 'category': 'Vegetables', 'cost': 25, 'margin': 0.35},
        
        # Premium Items (Low volume, high margin)
        {'id': 'G016', 'name': 'Ghee 500ml', 'category': 'Dairy', 'cost': 420, 'margin': 0.18},
        {'id': 'G017', 'name': 'Almonds 250g', 'category': 'Dry Fruits', 'cost': 680, 'margin': 0.22},
        {'id': 'G018', 'name': 'Honey 500g', 'category': 'Health', 'cost': 280, 'margin': 0.25},
        {'id': 'G019', 'name': 'Coffee Instant 100g', 'category': 'Beverages', 'cost': 150, 'margin': 0.28},
        {'id': 'G020', 'name': 'Pickle Mixed 400g', 'category': 'Condiments', 'cost': 95, 'margin': 0.32},
    ]
    
    for product in products:
        product['selling_price'] = round(product['cost'] * (1 + product['margin']), 2)
    
    return products

def get_day_multiplier(date, product_category):
    """Calculate sales multiplier based on day patterns"""
    day_of_week = date.weekday()  # 0=Monday, 6=Sunday
    day_of_month = date.day
    
    # Base multiplier
    multiplier = 1.0
    
    # Weekend boost (Saturday-Sunday) for most categories
    if day_of_week >= 5:  # Saturday or Sunday
        multiplier *= 1.4
    
    # Month beginning spike (1st-5th) - Salary days in India
    if day_of_month <= 5:
        multiplier *= 1.8
        
    # Mid-month slight boost (15th-18th) - Some companies pay mid-month
    elif 15 <= day_of_month <= 18:
        multiplier *= 1.3
        
    # Month end dip (26th-31st) - People waiting for next salary
    elif day_of_month >= 26:
        multiplier *= 0.7
    
    # Category-specific patterns
    if product_category == 'Staples':
        # Higher on month beginning
        if day_of_month <= 5:
            multiplier *= 1.2
    elif product_category == 'Dairy':
        # Daily consumption, less variance
        multiplier *= 0.9  # Reduce variance
    elif product_category == 'Snacks':
        # Higher on weekends
        if day_of_week >= 5:
            multiplier *= 1.3
    
    return multiplier

def generate_realistic_sales():
    """Generate sales data with realistic Indian grocery patterns"""
    products = generate_grocery_products()
    
    # Generate 4 months of data
    start_date = datetime(2025, 5, 1)
    end_date = datetime(2025, 8, 31)
    
    sales_data = []
    customer_id = 1
    
    current_date = start_date
    while current_date <= end_date:
        # Daily sales count varies
        day_multiplier = get_day_multiplier(current_date, 'General')
        base_daily_sales = random.randint(40, 80)
        daily_sales = int(base_daily_sales * day_multiplier)
        
        for _ in range(daily_sales):
            # Select product based on category weights
            category_weights = {
                'Staples': 30, 'Dairy': 15, 'Bakery': 10, 'Pulses': 8,
                'Cooking Oil': 5, 'Beverages': 8, 'Snacks': 8,
                'Personal Care': 6, 'Household': 4, 'Spices': 3,
                'Vegetables': 12, 'Protein': 7, 'Dry Fruits': 1,
                'Health': 2, 'Condiments': 3
            }
            
            # Select category
            categories = list(category_weights.keys())
            weights = list(category_weights.values())
            selected_category = np.random.choice(categories, p=np.array(weights)/sum(weights))
            
            # Select product from category
            category_products = [p for p in products if p['category'] == selected_category]
            if not category_products:
                continue
                
            product = random.choice(category_products)
            
            # Calculate quantity (most purchases are 1-2 units)
            if selected_category in ['Staples', 'Cooking Oil']:
                quantity = random.choices([1, 2, 3], weights=[50, 35, 15])[0]
            elif selected_category in ['Dairy', 'Bakery', 'Snacks']:
                quantity = random.choices([1, 2], weights=[70, 30])[0]
            else:
                quantity = 1
            
            # Apply day-specific multiplier for this product
            product_multiplier = get_day_multiplier(current_date, product['category'])
            
            # Price variation (±5%)
            price_variation = random.uniform(0.95, 1.05)
            final_price = round(product['selling_price'] * price_variation, 2)
            
            # Sales channel distribution
            channel = random.choices(
                ['retail', 'wholesale', 'online'], 
                weights=[85, 10, 5]
            )[0]
            
            sales_data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'product_id': product['id'],
                'product_name': product['name'],
                'quantity_sold': quantity,
                'unit_price': final_price,
                'customer_id': f'C{customer_id:04d}',
                'sales_channel': channel,
                'price': ''  # Keep empty for compatibility
            })
            
            customer_id += 1
            if customer_id > 9999:
                customer_id = 1
        
        current_date += timedelta(days=1)
    
    return sales_data, products

def generate_stock_data(products):
    """Generate realistic stock data for grocery products"""
    stock_data = []
    
    suppliers = {
        'Staples': 'S001', 'Pulses': 'S001', 'Cooking Oil': 'S002',
        'Dairy': 'S003', 'Bakery': 'S004', 'Protein': 'S003',
        'Beverages': 'S005', 'Snacks': 'S005', 'Personal Care': 'S006',
        'Household': 'S006', 'Spices': 'S007', 'Vegetables': 'S008',
        'Dry Fruits': 'S009', 'Health': 'S009', 'Condiments': 'S007'
    }
    
    for product in products:
        # Stock levels based on category
        if product['category'] in ['Staples', 'Dairy', 'Bakery']:
            current_stock = random.randint(40, 100)
            reorder_point = random.randint(20, 35)
            max_stock = random.randint(150, 300)
        elif product['category'] in ['Snacks', 'Personal Care', 'Beverages']:
            current_stock = random.randint(25, 60)
            reorder_point = random.randint(15, 25)
            max_stock = random.randint(100, 200)
        else:
            current_stock = random.randint(10, 30)
            reorder_point = random.randint(8, 15)
            max_stock = random.randint(50, 100)
        
        stock_data.append({
            'product_id': product['id'],
            'product_name': product['name'],
            'current_stock': current_stock,
            'reorder_point': reorder_point,
            'max_stock': max_stock,
            'unit_cost': product['cost'],
            'last_updated': '2025-08-31 10:00:00.000000',
            'category': product['category'],
            'supplier_id': suppliers.get(product['category'], 'S001')
        })
    
    return stock_data

def generate_suppliers_data():
    """Generate realistic supplier data for grocery business"""
    suppliers = [
        {'supplier_id': 'S001', 'name': 'Metro Wholesale Foods', 'contact': '+91-9876543210', 
         'email': 'orders@metrowholesale.com', 'category': 'Staples & Pulses', 'city': 'Mumbai'},
        {'supplier_id': 'S002', 'name': 'Golden Oil Distributors', 'contact': '+91-9876543211', 
         'email': 'sales@goldenoil.com', 'category': 'Cooking Oils', 'city': 'Delhi'},
        {'supplier_id': 'S003', 'name': 'Fresh Dairy Products', 'contact': '+91-9876543212', 
         'email': 'orders@freshdairy.com', 'category': 'Dairy & Protein', 'city': 'Pune'},
        {'supplier_id': 'S004', 'name': 'Daily Bread Bakery', 'contact': '+91-9876543213', 
         'email': 'supply@dailybread.com', 'category': 'Bakery Items', 'city': 'Bangalore'},
        {'supplier_id': 'S005', 'name': 'Snack Palace Distributors', 'contact': '+91-9876543214', 
         'email': 'orders@snackpalace.com', 'category': 'Snacks & Beverages', 'city': 'Chennai'},
        {'supplier_id': 'S006', 'name': 'Home Care Solutions', 'contact': '+91-9876543215', 
         'email': 'sales@homecare.com', 'category': 'Personal & Household', 'city': 'Hyderabad'},
        {'supplier_id': 'S007', 'name': 'Spice Garden Traders', 'contact': '+91-9876543216', 
         'email': 'orders@spicegarden.com', 'category': 'Spices & Condiments', 'city': 'Kochi'},
        {'supplier_id': 'S008', 'name': 'Farm Fresh Vegetables', 'contact': '+91-9876543217', 
         'email': 'supply@farmfresh.com', 'category': 'Fresh Vegetables', 'city': 'Nashik'},
        {'supplier_id': 'S009', 'name': 'Premium Health Foods', 'contact': '+91-9876543218', 
         'email': 'orders@premiumhealth.com', 'category': 'Dry Fruits & Health', 'city': 'Delhi'}
    ]
    return suppliers

if __name__ == "__main__":
    print("🏪 Generating Realistic Grocery Retail Data...")
    print("=" * 50)
    
    # Generate all data
    sales_data, products = generate_realistic_sales()
    stock_data = generate_stock_data(products)
    suppliers_data = generate_suppliers_data()
    
    # Save sales data
    sales_df = pd.DataFrame(sales_data)
    sales_df.to_csv('data/sales.csv', index=False)
    print(f"✅ Generated {len(sales_data)} sales transactions")
    
    # Save stock data
    stock_df = pd.DataFrame(stock_data)
    stock_df.to_csv('data/stock.csv', index=False)
    print(f"✅ Generated {len(stock_data)} product inventory records")
    
    # Save suppliers data
    suppliers_df = pd.DataFrame(suppliers_data)
    suppliers_df.to_csv('data/suppliers.csv', index=False)
    print(f"✅ Generated {len(suppliers_data)} supplier records")
    
    # Print summary statistics
    print("\n📊 DATA SUMMARY:")
    print(f"📅 Date Range: {sales_df['date'].min()} to {sales_df['date'].max()}")
    print(f"💰 Total Revenue: ₹{sales_df['unit_price'].sum():,.2f}")
    print(f"🛒 Average Order Value: ₹{sales_df['unit_price'].mean():.2f}")
    print(f"📦 Total Products: {len(products)}")
    print(f"🏢 Total Suppliers: {len(suppliers_data)}")
    
    # Show monthly pattern
    sales_df['date'] = pd.to_datetime(sales_df['date'])
    sales_df['month'] = sales_df['date'].dt.month
    monthly_sales = sales_df.groupby('month')['unit_price'].sum()
    print(f"\n📈 MONTHLY REVENUE PATTERN:")
    for month, revenue in monthly_sales.items():
        month_names = {5: 'May', 6: 'June', 7: 'July', 8: 'August'}
        print(f"   {month_names[month]}: ₹{revenue:,.2f}")
    
    # Show day-of-month pattern
    sales_df['day'] = sales_df['date'].dt.day
    sales_df['day_group'] = pd.cut(sales_df['day'], 
                                   bins=[0, 5, 15, 25, 31], 
                                   labels=['1-5 (Salary Days)', '6-15 (Mid-Month)', '16-25 (Late Month)', '26-31 (Month End)'])
    day_pattern = sales_df.groupby('day_group')['unit_price'].sum()
    print(f"\n📅 SALES PATTERN BY MONTH PERIOD:")
    for period, revenue in day_pattern.items():
        print(f"   {period}: ₹{revenue:,.2f}")
    
    print("\n🎯 Grocery retail data generation complete!")
    print("   Ready for AI-powered inventory management system!")
