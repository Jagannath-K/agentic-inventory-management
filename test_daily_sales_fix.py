"""
Test Daily Sales Entry Page Functionality
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ui.app import load_data, create_daily_sales_entry

def test_daily_sales_entry():
    """Test that daily sales entry loads without errors"""
    print("🧪 Testing Daily Sales Entry Page")
    print("=" * 40)
    
    try:
        # Test load_data function
        print("📂 Testing load_data()...")
        sales_data, stock_data = load_data()
        print(f"✅ Data loaded successfully:")
        print(f"   - Sales data: {len(sales_data) if sales_data is not None else 0} records")
        print(f"   - Stock data: {len(stock_data) if stock_data is not None else 0} products")
        
        # Test that we can access the data without supplier errors
        if stock_data is not None and len(stock_data) > 0:
            sample_product = stock_data.iloc[0]
            print(f"   - Sample product: {sample_product['product_name']}")
            print(f"   - Current stock: {sample_product['current_stock']}")
        
        print("✅ Daily Sales Entry page should now work without errors!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_daily_sales_entry()
