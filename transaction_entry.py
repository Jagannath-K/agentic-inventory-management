"""
Simple Web Interface for Daily Transaction Entry
Run this with: streamlit run transaction_entry.py
"""

import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime
import json

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.shop_operations import ShopOperations
from models.notification_system import NotificationSystem

def load_config():
    """Load system configuration"""
    config_file = 'config.json'
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    return {}

def main():
    st.set_page_config(
        page_title="Daily Transactions",
        page_icon="🛒",
        layout="wide"
    )
    
    st.title("🛒 Daily Transaction Entry")
    st.markdown("---")
    
    # Initialize components
    shop_ops = ShopOperations()
    notification_system = NotificationSystem()
    config = load_config()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Enter Transactions", "📊 Stock Status", "⚙️ Settings", "📧 Test Email"])
    
    with tab1:
        st.header("Enter Daily Transactions")
        
        # Load current stock for dropdown
        stock_summary = shop_ops.get_stock_summary()
        if stock_summary:
            product_options = {f"{item['product_id']} - {item['product_name']}": item['product_id'] 
                             for item in stock_summary}
        else:
            st.error("No stock data found. Please ensure stock.csv exists in the data folder.")
            return
        
        # Transaction form
        with st.form("transaction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                transaction_type = st.selectbox(
                    "Transaction Type",
                    ["Sale", "Purchase/Restock"]
                )
            
            with col2:
                selected_product = st.selectbox(
                    "Select Product",
                    options=list(product_options.keys())
                )
                product_id = product_options[selected_product]
            
            with col3:
                quantity = st.number_input(
                    "Quantity",
                    min_value=1,
                    value=1
                )
            
            # Additional fields for purchases
            if transaction_type == "Purchase/Restock":
                unit_cost = st.number_input(
                    "Unit Cost (optional)",
                    min_value=0.0,
                    value=0.0,
                    step=0.01
                )
            else:
                unit_cost = None
            
            # Submit button
            submitted = st.form_submit_button("Add Transaction")
            
            if submitted:
                # Create transaction
                transaction = {
                    'type': 'sale' if transaction_type == "Sale" else 'purchase',
                    'product_id': product_id,
                    'quantity': quantity,
                    'timestamp': datetime.now()
                }
                
                if unit_cost and unit_cost > 0:
                    transaction['unit_cost'] = unit_cost
                
                # Process transaction
                result = shop_ops.process_transactions([transaction])
                
                if result['success']:
                    st.success(f"✅ {transaction_type} recorded successfully!")
                    
                    # Show stock alerts if any
                    if result.get('stock_alerts'):
                        st.warning("⚠️ Stock alerts generated:")
                        for alert in result['stock_alerts']:
                            st.write(f"- {alert['product_name']}: {alert['current_stock']} units remaining (threshold: {alert['threshold']})")
                else:
                    st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    with tab2:
        st.header("Current Stock Status")
        
        # Refresh button
        if st.button("🔄 Refresh"):
            st.rerun()
        
        # Display stock
        stock_summary = shop_ops.get_stock_summary()
        if stock_summary:
            df = pd.DataFrame(stock_summary)
            
            # Color code based on stock levels
            def highlight_low_stock(row):
                threshold = config.get('inventory_settings', {}).get('default_low_stock_threshold', 10)
                if row['current_stock'] <= threshold:
                    return ['background-color: #ffcccb'] * len(row)
                return [''] * len(row)
            
            # Display the dataframe
            styled_df = df.style.apply(highlight_low_stock, axis=1)
            st.dataframe(styled_df, use_container_width=True)
            
            # Show summary stats
            col1, col2, col3, col4 = st.columns(4)
            
            threshold = config.get('inventory_settings', {}).get('default_low_stock_threshold', 10)
            low_stock_count = len([item for item in stock_summary if item['current_stock'] <= threshold])
            total_value = sum(item['current_stock'] * item['unit_cost'] for item in stock_summary)
            
            with col1:
                st.metric("Total Products", len(stock_summary))
            with col2:
                st.metric("Low Stock Items", low_stock_count)
            with col3:
                st.metric("Total Inventory Value", f"${total_value:,.2f}")
            with col4:
                avg_stock = sum(item['current_stock'] for item in stock_summary) / len(stock_summary)
                st.metric("Average Stock Level", f"{avg_stock:.1f}")
        else:
            st.warning("No stock data available.")
    
    with tab3:
        st.header("System Settings")
        
        # Load current config
        config = load_config()
        
        # Email settings
        st.subheader("📧 Email Notification Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            email_enabled = st.checkbox(
                "Enable Email Notifications",
                value=config.get('notification_settings', {}).get('email_enabled', True)
            )
        
        with col2:
            recipients = st.text_area(
                "Alert Recipients (one email per line)",
                value="\n".join(config.get('notification_settings', {}).get('stock_alert_recipients', ['manager@yourstore.com']))
            )
        
        # Threshold settings
        st.subheader("📊 Stock Threshold Settings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            low_threshold = st.number_input(
                "Low Stock Threshold",
                min_value=1,
                value=config.get('inventory_settings', {}).get('default_low_stock_threshold', 10)
            )
        
        with col2:
            critical_threshold = st.number_input(
                "Critical Stock Threshold",
                min_value=1,
                value=config.get('inventory_settings', {}).get('default_critical_stock_threshold', 5)
            )
        
        with col3:
            reorder_days = st.number_input(
                "Reorder Buffer Days",
                min_value=1,
                value=config.get('inventory_settings', {}).get('default_reorder_buffer_days', 7)
            )
        
        # Save settings
        if st.button("💾 Save Settings"):
            # Update config
            if 'notification_settings' not in config:
                config['notification_settings'] = {}
            if 'inventory_settings' not in config:
                config['inventory_settings'] = {}
            
            config['notification_settings']['email_enabled'] = email_enabled
            config['notification_settings']['stock_alert_recipients'] = [email.strip() for email in recipients.split('\n') if email.strip()]
            config['inventory_settings']['default_low_stock_threshold'] = low_threshold
            config['inventory_settings']['default_critical_stock_threshold'] = critical_threshold
            config['inventory_settings']['default_reorder_buffer_days'] = reorder_days
            
            # Save to file
            with open('config.json', 'w') as f:
                json.dump(config, f, indent=2)
            
            st.success("✅ Settings saved successfully!")
    
    with tab4:
        st.header("Test Email Configuration")
        
        test_email = st.text_input("Test Email Address", value="test@example.com")
        
        if st.button("📧 Send Test Email"):
            # Create test item
            test_item = {
                'product_id': 'TEST001',
                'product_name': 'Test Product',
                'current_stock': 5,
                'threshold': 10,
                'supplier_id': 'SUP001'
            }
            
            try:
                success = notification_system.send_stock_alert(
                    test_item,
                    [test_email],
                    "🧪 Test Stock Alert: {product_name}"
                )
                
                if success:
                    st.success("✅ Test email sent successfully!")
                else:
                    st.error("❌ Failed to send test email. Please check your email configuration in .env file.")
            except Exception as e:
                st.error(f"❌ Error: {e}")
        
        # Email configuration status
        st.subheader("Email Configuration Status")
        test_result = notification_system.test_email_configuration()
        
        if test_result['success']:
            st.success(f"✅ {test_result['message']}")
        else:
            st.error(f"❌ {test_result['error']}")
            st.info("💡 Please ensure EMAIL_USER and EMAIL_PASSWORD are set in your .env file")

if __name__ == "__main__":
    main()
