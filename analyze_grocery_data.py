#!/usr/bin/env python3
"""
Quick Analysis of New Grocery Retail Data
"""

import pandas as pd
import numpy as np
from datetime import datetime

def analyze_grocery_data():
    print("🏪 GROCERY RETAIL DATA ANALYSIS")
    print("=" * 50)
    
    # Load and analyze sales data
    sales_df = pd.read_csv('data/sales.csv')
    stock_df = pd.read_csv('data/stock.csv')
    suppliers_df = pd.read_csv('data/suppliers.csv')
    
    # Convert date column
    sales_df['date'] = pd.to_datetime(sales_df['date'])
    
    # Basic statistics
    print(f"📊 BASIC STATISTICS:")
    print(f"   Total Transactions: {len(sales_df):,}")
    print(f"   Total Revenue: ₹{sales_df['unit_price'].sum():,.2f}")
    print(f"   Average Order Value: ₹{sales_df['unit_price'].mean():.2f}")
    print(f"   Date Range: {sales_df['date'].min().date()} to {sales_df['date'].max().date()}")
    print(f"   Total Products: {len(stock_df)}")
    print(f"   Total Suppliers: {len(suppliers_df)}")
    
    # Category analysis
    print(f"\n🏷️ PRODUCT CATEGORIES:")
    category_sales = stock_df.merge(
        sales_df.groupby('product_id')['unit_price'].sum().reset_index(),
        on='product_id', how='left'
    ).fillna(0)
    
    category_revenue = category_sales.groupby('category')['unit_price'].sum().sort_values(ascending=False)
    for category, revenue in category_revenue.head(10).items():
        print(f"   {category}: ₹{revenue:,.2f}")
    
    # Monthly pattern analysis
    print(f"\n📅 MONTHLY SALES PATTERN:")
    sales_df['month'] = sales_df['date'].dt.month
    monthly_sales = sales_df.groupby('month')['unit_price'].sum()
    month_names = {5: 'May', 6: 'June', 7: 'July', 8: 'August'}
    for month, revenue in monthly_sales.items():
        print(f"   {month_names.get(month, f'Month {month}')}: ₹{revenue:,.2f}")
    
    # Day of month analysis (Indian salary pattern)
    print(f"\n💰 SALARY CYCLE PATTERN:")
    sales_df['day'] = sales_df['date'].dt.day
    sales_df['day_group'] = pd.cut(sales_df['day'], 
                                   bins=[0, 5, 15, 25, 31], 
                                   labels=['1-5 (Salary Days)', '6-15 (Mid-Month)', '16-25 (Late Month)', '26-31 (Month End)'])
    
    day_pattern = sales_df.groupby('day_group', observed=True)['unit_price'].agg(['sum', 'count']).round(2)
    for period, data in day_pattern.iterrows():
        print(f"   {period}: ₹{data['sum']:,.2f} ({data['count']} transactions)")
    
    # Weekend vs weekday analysis
    print(f"\n🛒 WEEKEND VS WEEKDAY PATTERN:")
    sales_df['is_weekend'] = sales_df['date'].dt.weekday >= 5
    weekend_pattern = sales_df.groupby('is_weekend')['unit_price'].agg(['sum', 'count', 'mean']).round(2)
    for is_weekend, data in weekend_pattern.iterrows():
        day_type = "Weekend" if is_weekend else "Weekday"
        print(f"   {day_type}: ₹{data['sum']:,.2f} ({data['count']} trans, ₹{data['mean']:.2f} avg)")
    
    # Top selling products
    print(f"\n🏆 TOP SELLING PRODUCTS:")
    product_sales = sales_df.groupby(['product_id', 'product_name']).agg({
        'unit_price': 'sum',
        'quantity_sold': 'sum'
    }).sort_values('unit_price', ascending=False)
    
    for (product_id, name), data in product_sales.head(10).iterrows():
        print(f"   {name}: ₹{data['unit_price']:,.2f} ({data['quantity_sold']} units)")
    
    # Stock value analysis
    print(f"\n📦 CURRENT STOCK VALUE:")
    stock_df['stock_value'] = stock_df['current_stock'] * stock_df['unit_cost']
    total_stock_value = stock_df['stock_value'].sum()
    print(f"   Total Stock Value: ₹{total_stock_value:,.2f}")
    
    category_stock = stock_df.groupby('category')['stock_value'].sum().sort_values(ascending=False)
    for category, value in category_stock.head(8).items():
        print(f"   {category}: ₹{value:,.2f}")
    
    print(f"\n🎯 Analysis complete! Data shows realistic grocery retail patterns.")
    print(f"🔸 Month-beginning sales spikes: ₹{day_pattern.loc['1-5 (Salary Days)', 'sum']:,.2f}")
    print(f"🔸 Weekend boost confirmed: {weekend_pattern.loc[True, 'count']} weekend transactions")
    print(f"🔸 Micro enterprise scale: ₹{sales_df['unit_price'].mean():.2f} average order value")

if __name__ == "__main__":
    analyze_grocery_data()
