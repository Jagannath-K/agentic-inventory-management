"""
Streamlit Dashboard for Agentic Inventory Management System
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os
import time

# Add parent directory to path to import agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from agents.planner import PlannerAgent
    from agents.executor import ExecutorAgent
    from agents.reflector import ReflectorAgent
    from models.predictor import DemandPredictor
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.error("Please ensure all agent modules are properly installed")

# Page configuration
st.set_page_config(
    page_title="Agentic Inventory Management",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        color: #262730;
    }
    .alert-critical {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #d32f2f;
    }
    .alert-warning {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #ef6c00;
    }
    .alert-success {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #2e7d32;
    }
    /* Fix only specific content areas without breaking navigation */
    .stExpander .streamlit-expanderContent {
        background-color: #ffffff;
        color: #262730;
    }
    .stTextArea textarea {
        color: #262730 !important;
        background-color: #ffffff !important;
    }
    /* Fix expander content specifically */
    .streamlit-expanderContent div {
        color: #262730 !important;
    }
    /* Fix only main page headers - not navigation */
    .main .block-container h1 {
        color: #1f77b4 !important;
        font-weight: 600 !important;
    }
    .main .block-container h2 {
        color: #1f77b4 !important;
        font-weight: 600 !important;
    }
    .main .block-container h3 {
        color: #1f77b4 !important;
        font-weight: 500 !important;
    }
    
    /* Improve dataframe table styling */
    .stDataFrame {
        background-color: white !important;
    }
    
    .stDataFrame table {
        background-color: white !important;
        border-collapse: collapse !important;
    }
    
    .stDataFrame thead th {
        background-color: #1f77b4 !important;
        color: white !important;
        font-weight: bold !important;
        text-align: center !important;
        padding: 12px 8px !important;
        border: 1px solid #ddd !important;
    }
    
    .stDataFrame tbody td {
        background-color: white !important;
        color: #262730 !important;
        padding: 10px 8px !important;
        border: 1px solid #ddd !important;
        text-align: left !important;
    }
    
    .stDataFrame tbody tr:nth-child(even) {
        background-color: #f8f9fa !important;
    }
    
    .stDataFrame tbody tr:hover {
        background-color: #e3f2fd !important;
    }
    
    /* Ensure text visibility in all dataframes */
    .stDataFrame div[data-testid="stDataFrame"] {
        background-color: white !important;
    }
    
    .stDataFrame div[data-testid="stDataFrame"] * {
        color: #262730 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agents_initialized' not in st.session_state:
    st.session_state.agents_initialized = False
    st.session_state.last_update = None

@st.cache_data
def load_data():
    """Load and cache data"""
    try:
        sales_data = pd.read_csv('data/sales.csv')
        stock_data = pd.read_csv('data/stock.csv')
        
        # Parse dates with modern pandas approach
        sales_data['date'] = pd.to_datetime(sales_data['date'], errors='coerce')
        stock_data['last_updated'] = pd.to_datetime(stock_data['last_updated'], errors='coerce')
        
        return sales_data, stock_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

def render_delivery_tracking_ui():
    """Render the delivery tracking and confirmation interface"""
    st.markdown('<h2 style="color: #1f77b4; font-weight: 600;">📦 Delivery Management</h2>', unsafe_allow_html=True)
    
    # Ensure executor is available
    if 'executor' not in st.session_state or st.session_state.executor is None:
        if not initialize_agents():
            st.error("Failed to initialize agents")
            return
    
    executor = st.session_state.executor
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["🚚 Pending Deliveries", "✅ Confirm Delivery", "📋 Order History"])
    
    with tab1:
        st.markdown('<h3 style="color: #1f77b4; font-weight: 500;">Pending Deliveries</h3>', unsafe_allow_html=True)
        
        # Get pending orders
        try:
            pending_orders = executor.process("get_pending_orders")
            
            if pending_orders:
                # Convert to DataFrame for better display
                df_pending = pd.DataFrame(pending_orders)
                
                # Format the display
                display_df = df_pending[['order_id', 'product_name', 'quantity_ordered', 
                                       'cost', 'priority', 'expected_delivery']].copy()
                display_df.columns = ['Order ID', 'Product', 'Quantity', 'Cost (₹)', 'Priority', 'Expected Delivery']
                
                # Format cost with Indian rupee symbol
                display_df['Cost (₹)'] = display_df['Cost (₹)'].apply(lambda x: f"₹{x:,.2f}")
                
                # Convert expected delivery to readable format
                display_df['Expected Delivery'] = pd.to_datetime(display_df['Expected Delivery']).dt.strftime('%Y-%m-%d')
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Pending Orders", len(pending_orders))
                with col2:
                    total_value = sum([order['cost'] for order in pending_orders])
                    st.metric("Total Value", f"₹{total_value:,.2f}")
                with col3:
                    high_priority = len([o for o in pending_orders if o['priority'] in ['CRITICAL', 'HIGH']])
                    st.metric("High Priority", high_priority)
                    
            else:
                st.success("✅ No pending deliveries. All orders have been received!")
                
        except Exception as e:
            st.error(f"Error loading pending orders: {e}")
    
    with tab2:
        st.markdown('<h3 style="color: #1f77b4; font-weight: 500;">Confirm Delivery</h3>', unsafe_allow_html=True)
        
        try:
            # Get pending orders for selection
            pending_orders = executor.process("get_pending_orders")
            
            if pending_orders:
                # Select order to confirm
                order_options = {f"{order['order_id']} - {order['product_name']} ({order['quantity_ordered']} units)": 
                               order['order_id'] for order in pending_orders}
                
                selected_order_display = st.selectbox(
                    "Select order to confirm delivery:",
                    options=list(order_options.keys()),
                    key="delivery_order_select"
                )
                
                if selected_order_display:
                    selected_order_id = order_options[selected_order_display]
                    
                    # Find the order details
                    selected_order = next(order for order in pending_orders if order['order_id'] == selected_order_id)
                    
                    # Order details display
                    st.markdown("**Order Details:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"• **Product:** {selected_order['product_name']}")
                        st.write(f"• **Ordered Quantity:** {selected_order['quantity_ordered']} units")
                        st.write(f"• **Order Date:** {selected_order['order_date'][:10]}")
                    with col2:
                        st.write(f"• **Expected Delivery:** {selected_order['expected_delivery'][:10]}")
                        st.write(f"• **Priority:** {selected_order['priority']}")
                        st.write(f"• **Cost:** ₹{selected_order['cost']:,.2f}")
                    
                    st.markdown("---")
                    
                    # Delivery confirmation form
                    st.markdown("**Confirm Delivery:**")
                    
                    # Received quantity input
                    max_quantity = selected_order['quantity_ordered']
                    received_quantity = st.number_input(
                        f"Received Quantity (max: {max_quantity})",
                        min_value=1,
                        max_value=max_quantity,
                        value=max_quantity,
                        help="Enter the actual quantity received from supplier",
                        key="delivery_quantity"
                    )
                    
                    # Confirm button
                    if st.button("✅ Confirm Delivery", type="primary", key="confirm_delivery_btn"):
                        # Process delivery confirmation
                        with st.spinner("Confirming delivery..."):
                            try:
                                result = executor.process("confirm_delivery", 
                                                        order_id=selected_order_id, 
                                                        received_quantity=received_quantity)
                                
                                if result['success']:
                                    st.success(f"✅ Delivery confirmed successfully!")
                                    st.success(f"📦 {result['product_name']}: Stock updated from {result['old_stock']} to {result['new_stock']} units")
                                    
                                    # Show partial delivery warning
                                    if received_quantity < max_quantity:
                                        missing = max_quantity - received_quantity
                                        st.warning(f"⚠️ Partial delivery: {missing} units still missing. Consider placing a new order.")
                                    
                                    # Clear the form
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error(f"❌ Failed to confirm delivery: {result['message']}")
                            except Exception as e:
                                st.error(f"Error confirming delivery: {e}")
            else:
                st.info("No pending orders to confirm.")
                
        except Exception as e:
            st.error(f"Error in delivery confirmation: {e}")
    
    with tab3:
        st.markdown('<h3 style="color: #1f77b4; font-weight: 500;">Order History</h3>', unsafe_allow_html=True)
        
        try:
            # Show completed orders
            completed_orders = executor.completed_orders
            
            if completed_orders:
                df_completed = pd.DataFrame(completed_orders)
                
                # Format for display
                display_completed = df_completed[['order_id', 'product_name', 'quantity_ordered', 
                                               'received_quantity', 'cost', 'delivery_date']].copy()
                display_completed.columns = ['Order ID', 'Product', 'Ordered', 'Received', 'Cost (₹)', 'Delivered']
                
                # Format cost
                display_completed['Cost (₹)'] = display_completed['Cost (₹)'].apply(lambda x: f"₹{x:,.2f}")
                
                # Format delivery date
                display_completed['Delivered'] = pd.to_datetime(display_completed['Delivered']).dt.strftime('%Y-%m-%d %H:%M')
                
                st.dataframe(display_completed, use_container_width=True, hide_index=True)
                
                # Summary
                total_completed = len(completed_orders)
                total_value_completed = sum([order['cost'] for order in completed_orders])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Completed Orders", total_completed)
                with col2:
                    st.metric("Total Value Received", f"₹{total_value_completed:,.2f}")
            else:
                st.info("No completed orders yet.")
                
        except Exception as e:
            st.error(f"Error loading order history: {e}")

def initialize_agents():
    """Initialize all agents"""
    if not st.session_state.agents_initialized:
        with st.spinner("Initializing AI agents..."):
            try:
                st.session_state.planner = PlannerAgent()
                st.session_state.executor = ExecutorAgent()
                st.session_state.reflector = ReflectorAgent()
                st.session_state.predictor = DemandPredictor()
                st.session_state.agents_initialized = True
                st.success("AI agents initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing agents: {e}")
                return False
    return True

def create_overview_dashboard():
    """Create overview dashboard"""
    st.markdown('<h1 class="main-header">📦 Agentic Inventory Management System</h1>', unsafe_allow_html=True)
    
    # Load data
    sales_data, stock_data = load_data()
    
    if sales_data is None:
        st.error("Unable to load data. Please check data files.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_products = len(stock_data)
        st.metric("Total Products", total_products)
    
    with col2:
        total_inventory_value = (stock_data['current_stock'] * stock_data['unit_cost']).sum()
        st.metric("Inventory Value", f"₹{total_inventory_value:,.2f}")
    
    with col3:
        low_stock_count = len(stock_data[stock_data['current_stock'] <= stock_data['reorder_point']])
        st.metric("Low Stock Items", low_stock_count, delta=f"{low_stock_count/total_products*100:.1f}%")
    
    with col4:
        recent_sales = sales_data[sales_data['date'] >= datetime.now() - timedelta(days=7)]
        weekly_revenue = (recent_sales['quantity_sold'] * recent_sales['unit_price']).sum()
        st.metric("Weekly Revenue", f"₹{weekly_revenue:,.2f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sales Trend (Last 30 Days)")
        recent_sales = sales_data[sales_data['date'] >= datetime.now() - timedelta(days=30)]
        daily_sales = recent_sales.groupby('date').agg({
            'quantity_sold': 'sum',
            'unit_price': 'mean'
        }).reset_index()
        daily_sales['revenue'] = daily_sales['quantity_sold'] * daily_sales['unit_price']
        
        fig = px.line(daily_sales, x='date', y='revenue', title="Daily Revenue")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Stock Status Distribution")
        stock_status = []
        for _, row in stock_data.iterrows():
            if row['current_stock'] <= row['reorder_point'] * 0.5:
                status = 'Critical'
            elif row['current_stock'] <= row['reorder_point']:
                status = 'Low'
            elif row['current_stock'] >= row['max_stock'] * 0.9:
                status = 'Overstocked'
            else:
                status = 'Normal'
            stock_status.append(status)
        
        status_counts = pd.Series(stock_status).value_counts()
        fig = px.pie(values=status_counts.values, names=status_counts.index, 
                    title="Stock Status Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity
    st.subheader("Recent Activity")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Top Selling Products (Last 7 Days)**")
        recent_sales = sales_data[sales_data['date'] >= datetime.now() - timedelta(days=7)]
        top_products = recent_sales.groupby('product_name')['quantity_sold'].sum().sort_values(ascending=False).head(5)
        
        for product, quantity in top_products.items():
            st.write(f"• {product}: {quantity} units")
    
    with col2:
        st.write("**Low Stock Alerts**")
        low_stock_items = stock_data[stock_data['current_stock'] <= stock_data['reorder_point']].head(5)
        
        for _, item in low_stock_items.iterrows():
            urgency = "🔴" if item['current_stock'] <= item['reorder_point'] * 0.5 else "🟡"
            st.write(f"{urgency} {item['product_name']}: {item['current_stock']} units")

def create_ai_planning_dashboard():
    """Create AI planning dashboard"""
    st.markdown('<h1 style="color: #1f77b4; font-weight: 600;">🤖 AI Planning Agent</h1>', unsafe_allow_html=True)
    
    if not initialize_agents():
        return
    
    # Planning controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<h2 style="color: #1f77b4; font-weight: 600;">Inventory Planning</h2>', unsafe_allow_html=True)
        
    with col2:
        if st.button("Generate New Plan", type="primary"):
            with st.spinner("AI agent is analyzing patterns and creating optimal inventory plan..."):
                plans = st.session_state.planner.create_inventory_plan()
                st.session_state.current_plans = plans
                st.session_state.last_update = datetime.now()
                st.success(f"Generated plan for {len(plans)} products!")
    
    # Add visual guide
    st.info("💡 **Table Color Guide:** 🔴 Critical = Immediate reorder needed | 🟡 High = Reorder soon | 🟠 Medium = Monitor closely | 🟢 Low = Stock sufficient")
    
    # Display plans if available
    if 'current_plans' in st.session_state and st.session_state.current_plans:
        plans = st.session_state.current_plans
        
        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            critical_count = len([p for p in plans if p.urgency_level == 'CRITICAL'])
            st.metric("Critical Items", critical_count)
        
        with col2:
            high_count = len([p for p in plans if p.urgency_level == 'HIGH'])
            st.metric("High Priority", high_count)
        
        with col3:
            # Count of actionable items (excluding LOW urgency)
            actionable_count = len([p for p in plans if p.urgency_level in ['CRITICAL', 'HIGH', 'MEDIUM']])
            st.metric("Actionable Items", actionable_count, 
                     help="Items needing immediate to medium-term attention")
        
        with col4:
            # Calculate value only for items that actually need reordering (excluding LOW urgency)
            actionable_plans = [p for p in plans if p.urgency_level in ['CRITICAL', 'HIGH', 'MEDIUM'] and p.reorder_quantity > 0]
            total_reorder_value = sum(p.reorder_quantity * 
                                    st.session_state.planner.stock_data[
                                        st.session_state.planner.stock_data['product_id'] == p.product_id
                                    ]['unit_cost'].iloc[0] for p in actionable_plans)
            st.metric("Reorder Investment", f"₹{total_reorder_value:,.2f}", 
                     help="Investment needed for critical, high & medium priority items")
        
        with col5:
            avg_confidence = np.mean([p.confidence_score for p in plans])
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        # Detailed plans table
        st.markdown('<h3 style="color: #1f77b4; font-weight: 500;">Detailed Inventory Plans</h3>', unsafe_allow_html=True)
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            urgency_filter = st.selectbox("Filter by Urgency", 
                                        ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
        with col2:
            show_only_reorder = st.checkbox("Show only items needing reorder")
        
        # Filter plans
        filtered_plans = plans
        if urgency_filter != "All":
            filtered_plans = [p for p in filtered_plans if p.urgency_level == urgency_filter]
        if show_only_reorder:
            filtered_plans = [p for p in filtered_plans if p.reorder_quantity > 0]
        
        # Create DataFrame for display
        plan_data = []
        for plan in filtered_plans:
            # Add visual indicators for urgency
            urgency_display = plan.urgency_level
            if plan.urgency_level == 'CRITICAL':
                urgency_display = '🔴 CRITICAL'
            elif plan.urgency_level == 'HIGH':
                urgency_display = '🟡 HIGH'
            elif plan.urgency_level == 'MEDIUM':
                urgency_display = '🟠 MEDIUM'
            else:
                urgency_display = '🟢 LOW'
            
            plan_data.append({
                'Product': plan.product_name,
                'Current Stock': plan.current_stock,
                'Predicted Demand (30d)': f"{plan.predicted_demand:.1f}",
                'Reorder Quantity': plan.reorder_quantity,
                'Urgency': urgency_display,
                'Reorder Date': plan.reorder_date.strftime('%Y-%m-%d'),
                'Confidence': f"{plan.confidence_score:.1%}",
                'Reasoning': plan.reasoning  # Show full reasoning text
            })
        
        if plan_data:
            df = pd.DataFrame(plan_data)
            
            # Improved color coding with better contrast
            def highlight_urgency(row):
                urgency = row['Urgency']
                if 'CRITICAL' in urgency:
                    return ['background-color: #ffcdd2; color: #b71c1c; font-weight: bold'] * len(row)
                elif 'HIGH' in urgency:
                    return ['background-color: #fff3e0; color: #e65100; font-weight: bold'] * len(row)
                elif 'MEDIUM' in urgency:
                    return ['background-color: #fff8e1; color: #f57c00'] * len(row)
                else:
                    return ['background-color: #e8f5e8; color: #1b5e20'] * len(row)
            
            # Apply styling with better contrast and text wrapping
            styled_df = df.style.apply(highlight_urgency, axis=1)
            styled_df = styled_df.set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#1f77b4'), 
                                             ('color', 'white'), 
                                             ('font-weight', 'bold'),
                                             ('text-align', 'center')]},
                {'selector': 'td', 'props': [('text-align', 'left'),
                                             ('padding', '8px'),
                                             ('border', '1px solid #ddd'),
                                             ('white-space', 'pre-wrap'),
                                             ('word-wrap', 'break-word'),
                                             ('max-width', '300px')]},
                {'selector': 'table', 'props': [('border-collapse', 'collapse'),
                                               ('width', '100%')]},
                {'selector': 'td:nth-child(8)', 'props': [('max-width', '400px'),
                                                          ('min-width', '200px')]}  # Reasoning column
            ])
            
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Add expandable detailed view for reasoning
            with st.expander("📋 View Detailed Reasoning for All Items"):
                for i, plan in enumerate(filtered_plans, 1):
                    urgency_emoji = {
                        'CRITICAL': '🔴', 'HIGH': '🟡', 'MEDIUM': '🟠', 'LOW': '🟢'
                    }.get(plan.urgency_level, '⚪')
                    
                    st.markdown(f"**{i}. {plan.product_name}** (Stock: {plan.current_stock}, Urgency: {urgency_emoji} {plan.urgency_level})")
                    st.markdown(f"*{plan.reasoning}*")
                    st.markdown("---")
            
            # Export option
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Plan as CSV",
                data=csv,
                file_name=f"inventory_plan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime='text/csv'
            )
        else:
            st.info("No plans match the selected filters.")
            
        # Add execution section within AI Planning
        st.markdown("---")
        st.markdown('<h2 style="color: #1f77b4; font-weight: 600;">⚡ Order Execution</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown('<h3 style="color: #1f77b4; font-weight: 500;">Execute Orders Based on Predicted Demand</h3>', unsafe_allow_html=True)
            
        with col2:
            if st.button("Execute Orders", type="primary"):
                with st.spinner("AI agent is executing orders based on predicted demand..."):
                    try:
                        execution_results = st.session_state.executor.process("execute_plans", plans=plans)
                        st.session_state.execution_results = execution_results
                        st.success("Order execution completed! Orders sent to supplier.")
                    except Exception as e:
                        st.error(f"Error during execution: {e}")
        
        # Display execution results
        if 'execution_results' in st.session_state:
            results = st.session_state.execution_results
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Plans", results['total_plans'])
            
            with col2:
                st.metric("Successful Orders", results['successful_orders'])
            
            with col3:
                st.metric("Failed Orders", results['failed_orders'])
            
            with col4:
                st.metric("Total Cost", f"₹{results['total_cost']:,.2f}")
            
            # Detailed execution results
            st.markdown('<h3 style="color: #1f77b4; font-weight: 500;">Execution Details</h3>', unsafe_allow_html=True)
            
            if results['order_details']:
                execution_data = []
                for order in results['order_details']:
                    status_icon = "✅" if order['success'] else "❌"
                    execution_data.append({
                        'Status': status_icon,
                        'Product': order['product_name'],
                        'Order ID': order['order_id'],
                        'Cost': f"₹{order.get('cost', 0):,.2f}",
                        'Message': order['message']
                    })
                
                df = pd.DataFrame(execution_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No order details available.")
        
        # Add Delivery Tracking UI below Order Execution
        st.markdown("---")
        render_delivery_tracking_ui()
    
    else:
        st.info("Click 'Generate New Plan' to create an AI-powered inventory plan.")


def create_analytics_dashboard():
    """Create analytics and reflection dashboard"""
    st.markdown('<h1 style="color: #1f77b4; font-weight: 600;">📊 AI Analytics & Insights</h1>', unsafe_allow_html=True)
    
    if not initialize_agents():
        return
    
    # Analytics controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("System Performance Analysis")
        
    with col2:
        # Quick stock status preview
        try:
            from models.shop_operations import ShopOperations
            shop_ops = ShopOperations()
            low_stock_items = shop_ops.get_low_stock_items()
            critical_items = [item for item in low_stock_items if item['current_stock'] <= 5]
            
            if low_stock_items:
                if critical_items:
                    st.warning(f"🚨 {len(critical_items)} critical, {len(low_stock_items)} total alerts pending")
                else:
                    st.info(f"📧 {len(low_stock_items)} stock alerts pending")
            else:
                st.success("✅ All stock levels good")
        except:
            pass  # Don't break the UI if stock check fails
            
        if st.button("Generate Report and Alert", type="primary"):
            with st.spinner("AI agent is analyzing system performance and checking stock alerts..."):
                try:
                    # Generate analytics report
                    report = st.session_state.reflector.create_optimization_report()
                    st.session_state.analytics_report = report
                    
                    # Check and send stock alerts automatically
                    from models.shop_operations import ShopOperations
                    from models.notification_system import NotificationSystem
                    
                    # Initialize components for stock checking
                    shop_ops = ShopOperations()
                    notification_system = NotificationSystem()
                    
                    # Get low stock items using reorder points
                    low_stock_items = shop_ops.get_low_stock_items()
                    
                    alerts_sent = 0
                    critical_alerts_sent = 0
                    
                    if low_stock_items:
                        st.info(f"📧 Found {len(low_stock_items)} items requiring stock alerts...")
                        
                        for item in low_stock_items:
                            try:
                                success = notification_system.send_stock_alert(
                                    item,
                                    [],  # Recipients configured in notification system
                                    "🚨 Stock Alert: {product_name} is running low!"
                                )
                                if success:
                                    alerts_sent += 1
                                    
                                # Check if it's also critical (below critical threshold)
                                critical_threshold = 5  # You can adjust this value
                                if item['current_stock'] <= critical_threshold:
                                    critical_success = notification_system.send_critical_stock_alert(item)
                                    if critical_success:
                                        critical_alerts_sent += 1
                                        
                            except Exception as e:
                                st.warning(f"Alert failed for {item['product_name']}: {str(e)}")
                    
                    # Display results
                    success_message = "✅ Analytics report generated!"
                    if alerts_sent > 0:
                        success_message += f"\n📧 {alerts_sent} stock alerts sent to registered recipients"
                    if critical_alerts_sent > 0:
                        success_message += f"\n🚨 {critical_alerts_sent} critical alerts sent!"
                    if alerts_sent == 0:
                        success_message += "\n✨ No stock alerts needed - all items are well stocked!"
                    
                    st.success(success_message)
                    
                except Exception as e:
                    st.error(f"Error generating report or sending alerts: {e}")
    
    # Display analytics if available
    if 'analytics_report' in st.session_state:
        report = st.session_state.analytics_report
        
        # System health overview
        health_color = {
            'EXCELLENT': 'green',
            'GOOD': 'blue',
            'NEEDS_ATTENTION': 'orange',
            'CRITICAL': 'red'
        }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            health = report['summary']['overall_system_health']
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: {health_color.get(health, 'black')}">System Health</h3>
                <h2>{health}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Products Analyzed", report['summary']['total_products_analyzed'])
        
        with col3:
            st.metric("High Priority Issues", report['summary']['high_priority_issues'])
        
        # Key Performance Indicators
        st.subheader("Key Performance Indicators")
        
        kpi_data = []
        for kpi in report['key_performance_indicators']:
            status_icon = "🔴" if kpi['priority'] == 'CRITICAL' else "🟡" if kpi['priority'] == 'HIGH' else "🟢"
            kpi_data.append({
                'Status': status_icon,
                'Metric': kpi['metric'],
                'Current': f"{kpi['current']:.2f}",
                'Target': f"{kpi['target']:.2f}",
                'Trend': kpi['trend'],
                'Priority': kpi['priority'],
                'Suggestion': kpi['suggestion']
            })
        
        if kpi_data:
            df = pd.DataFrame(kpi_data)
            st.dataframe(df, use_container_width=True)
        
        # Critical Insights
        st.subheader("Critical Insights")
        
        for insight in report['critical_insights']:
            impact_color = {
                'HIGH': 'alert-critical',
                'MEDIUM': 'alert-warning',
                'LOW': 'alert-success'
            }
            
            alert_class = impact_color.get(insight['impact'], 'alert-success')
            
            st.markdown(f"""
            <div class="{alert_class}">
                <h4 style="color: inherit; margin: 0 0 10px 0;">{insight['title']} ({insight['impact']} Impact)</h4>
                <p style="color: inherit; margin: 5px 0;">{insight['description']}</p>
                <p style="color: inherit; margin: 5px 0;"><strong>Estimated Benefit:</strong> {insight['benefit']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("Recommended Actions", expanded=False):
                for action in insight['actions']:
                    st.markdown(f"""
                    <div style="color: #262730; background-color: #ffffff; padding: 5px;">
                        • {action}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Recommendations
        st.subheader("Prioritized Recommendations")
        
        for i, rec in enumerate(report['recommendations'], 1):
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 3px solid #007bff;">
                <span style="color: #262730; font-weight: bold;">{i}.</span> 
                <span style="color: #262730;">{rec}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Stockout Risk Analysis
        if report['stockout_risks']:
            st.subheader("Stockout Risk Analysis")
            
            risk_data = []
            for product_id, risk_info in report['stockout_risks'].items():
                risk_emoji = "🔴" if risk_info['risk_level'] == 'CRITICAL' else "🟡"
                risk_data.append({
                    'Risk': risk_emoji,
                    'Product ID': product_id,
                    'Current Stock': risk_info['current_stock'],
                    'Daily Velocity': f"{risk_info['daily_velocity']:.1f}",
                    'Days Until Stockout': f"{risk_info['days_until_stockout']:.1f}",
                    'Risk Level': risk_info['risk_level']
                })
            
            if risk_data:
                df = pd.DataFrame(risk_data)
                st.dataframe(df, use_container_width=True)

def create_demand_forecasting_dashboard():
    """Create demand forecasting dashboard"""
    st.markdown('<h1 style="color: #1f77b4; font-weight: 600;">🔮 AI Demand Forecasting</h1>', unsafe_allow_html=True)
    
    if not initialize_agents():
        return
    
    # Load data
    sales_data, stock_data = load_data()
    
    if stock_data is None:
        return
    
    # Product selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create a mapping of product names to IDs
        product_options = stock_data['product_name'].tolist()
        selected_product_name = st.selectbox("Select Product for Forecasting", 
                                           product_options)
        # Get the product_id for the selected product name
        selected_product = stock_data[stock_data['product_name'] == selected_product_name]['product_id'].iloc[0]
    
    with col2:
        forecast_days = st.number_input("Forecast Days", min_value=1, max_value=90, value=30)
    
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("AI is analyzing patterns and generating forecast..."):
            try:
                # Train model for selected product
                training_result = st.session_state.predictor.train_models(selected_product)
                
                # Generate predictions
                predictions = []
                dates = []
                
                for day in range(1, forecast_days + 1):
                    target_date = datetime.now() + timedelta(days=day)
                    pred_result = st.session_state.predictor.predict_demand(selected_product, target_date)
                    predictions.append(pred_result['predicted_demand'])
                    dates.append(target_date)
                
                # Store results
                st.session_state.forecast_results = {
                    'product_id': selected_product,
                    'dates': dates,
                    'predictions': predictions,
                    'training_result': training_result
                }
                
                st.success("Forecast generated successfully!")
                
            except Exception as e:
                st.error(f"Error generating forecast: {e}")
    
    # Display forecast results
    if 'forecast_results' in st.session_state:
        results = st.session_state.forecast_results
        
        if results['product_id'] == selected_product:
            # Get product name for display
            product_display_name = stock_data[stock_data['product_id'] == selected_product]['product_name'].iloc[0]
            
            # Forecast chart
            st.subheader(f"Demand Forecast for {product_display_name}")
            
            # Get historical data for context
            product_sales = sales_data[sales_data['product_id'] == selected_product]
            
            if not product_sales.empty:
                # Historical sales
                historical = product_sales.groupby('date')['quantity_sold'].sum().reset_index()
                historical = historical.sort_values('date')
                
                # Create combined chart
                fig = make_subplots(specs=[[{"secondary_y": False}]])
                
                # Historical data
                fig.add_trace(
                    go.Scatter(x=historical['date'], y=historical['quantity_sold'],
                              mode='lines+markers', name='Historical Sales',
                              line=dict(color='blue'))
                )
                
                # Forecast data
                fig.add_trace(
                    go.Scatter(x=results['dates'], y=results['predictions'],
                              mode='lines+markers', name='Forecast',
                              line=dict(color='red', dash='dash'))
                )
                
                fig.update_layout(title=f"Sales History and Forecast for {product_display_name}",
                                xaxis_title="Date", yaxis_title="Quantity")
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Forecast summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_forecast = sum(results['predictions'])
                st.metric("Total Forecasted Demand", f"{total_forecast:.0f} units")
            
            with col2:
                avg_daily = np.mean(results['predictions'])
                st.metric("Average Daily Demand", f"{avg_daily:.1f} units")
            
            with col3:
                max_daily = max(results['predictions'])
                st.metric("Peak Daily Demand", f"{max_daily:.0f} units")
            
            # Model performance
            if results['training_result']:
                st.subheader("Model Performance")
                
                performance = results['training_result'].get('performance', {})
                if performance:
                    perf_data = []
                    for model, metrics in performance.items():
                        perf_data.append({
                            'Model': model,
                            'Test MAE': f"{metrics['test_mae']:.2f}",
                            'Test RMSE': f"{metrics['test_rmse']:.2f}",
                            'Test R²': f"{metrics['test_r2']:.3f}"
                        })
                    
                    if perf_data:
                        df = pd.DataFrame(perf_data)
                        st.dataframe(df, use_container_width=True)
                
                best_model = results['training_result'].get('best_model')
                if best_model:
                    st.success(f"Best performing model: {best_model}")

def create_daily_sales_entry():
    """Create daily sales entry interface for grocery store owner"""
    st.markdown('<h1 class="main-header">🛒 Daily Sales & Stock Management</h1>', unsafe_allow_html=True)
    st.markdown("**Manage your daily sales and stock refills to keep inventory updated and improve AI predictions**")
    
    # Load current data
    sales_data, stock_data = load_data()
    
    if stock_data is None:
        st.error("Unable to load stock data. Please check data files.")
        return
    
    # Create tabs for different entry methods
    tab1, tab2, tab3 = st.tabs(["📝 Sales Entry", "📊 Today's Summary", "📦 Stock Refill"])
    
    with tab1:
        st.subheader("Sales Entry")
        
        # Initialize session state for bulk entries
        if 'bulk_entries' not in st.session_state:
            st.session_state.bulk_entries = []
        
        col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
        
        with col1:
            bulk_product = st.selectbox(
                "Product",
                options=stock_data['product_name'].tolist(),
                key="bulk_product"
            )
        
        with col2:
            bulk_qty = st.number_input("Qty", min_value=1, value=1, key="bulk_qty")
        
        with col3:
            product_info = stock_data[stock_data['product_name'] == bulk_product].iloc[0]
            fixed_price = product_info['unit_cost']
            st.number_input("Price ₹ (Fixed)", min_value=0.0, value=float(fixed_price), key="bulk_price", disabled=True, help="Prices are fixed and cannot be changed")
            bulk_price = fixed_price  
        with col4:
            if st.button("➕ Add", key="add_bulk"):
                product_info = stock_data[stock_data['product_name'] == bulk_product].iloc[0]
                st.session_state.bulk_entries.append({
                    'product_id': product_info['product_id'],
                    'product_name': bulk_product,
                    'quantity': bulk_qty,
                    'price': bulk_price,
                    'total': bulk_qty * bulk_price
                })
                st.rerun()
        
        # Display bulk entries
        if st.session_state.bulk_entries:
            st.markdown("#### Pending Sales Entries")
            bulk_df = pd.DataFrame(st.session_state.bulk_entries)
            st.dataframe(bulk_df)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("💰 Process All Sales", type="primary"):
                    success_count = 0
                    for entry in st.session_state.bulk_entries:
                        if add_sale_to_system(
                            entry['product_id'], entry['product_name'], 
                            entry['quantity'], entry['price'],
                            f"C{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        ):
                            success_count += 1
                    
                    st.success(f"✅ Processed {success_count} sales!")
                    st.session_state.bulk_entries = []
                    st.cache_data.clear()
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Clear All"):
                    st.session_state.bulk_entries = []
                    st.rerun()
            
            with col3:
                total_value = sum(entry['total'] for entry in st.session_state.bulk_entries)
                st.metric("Total Value", f"₹{total_value:.2f}")
    
    with tab2:
        st.subheader("Today's Sales Summary")
        
        # Load today's sales
        today = datetime.now().date()
        if sales_data is not None and len(sales_data) > 0:
            sales_data['date'] = pd.to_datetime(sales_data['date']).dt.date
            today_sales = sales_data[sales_data['date'] == today]
            
            if len(today_sales) > 0:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Today's Transactions", len(today_sales))
                
                with col2:
                    total_revenue = (today_sales['quantity_sold'] * today_sales['unit_price']).sum()
                    st.metric("Total Revenue", f"₹{total_revenue:.2f}")
                
                with col3:
                    st.metric("Items Sold", today_sales['quantity_sold'].sum())
                
                with col4:
                    avg_transaction = total_revenue / len(today_sales) if len(today_sales) > 0 else 0
                    st.metric("Avg Transaction", f"₹{avg_transaction:.2f}")
                
                # Today's sales by product
                st.markdown("#### Today's Sales by Product")
                product_sales = today_sales.groupby('product_name').agg({
                    'quantity_sold': 'sum',
                    'unit_price': 'first'  
                }).round(2)
                
                if len(product_sales) > 0:
                    # Calculate correct revenue: quantity × unit_price
                    product_sales['Revenue'] = product_sales['quantity_sold'] * product_sales['unit_price']
                    product_sales['Quantity'] = product_sales['quantity_sold']
                    st.dataframe(
                        product_sales[['Quantity', 'Revenue']].sort_values('Revenue', ascending=False),
                        use_container_width=True
                    )
                
                # Sales chart
                if len(today_sales) > 5:
                    fig = px.bar(
                        product_sales.reset_index(),
                        x='product_name',
                        y='Revenue',
                        title="Today's Revenue by Product",
                        labels={'Revenue': 'Revenue (₹)', 'product_name': 'Product'}
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.info("No sales recorded for today yet. Start adding sales using the Sales Entry tab!")
        else:
            st.info("No sales data available. Start recording your first sale!")

    with tab3:
        st.subheader("📦 Stock Refill Management")
        st.markdown("**Add new stock when you receive deliveries from suppliers**")
        
        # Initialize session state for stock refill entries
        if 'refill_entries' not in st.session_state:
            st.session_state.refill_entries = []
        
        col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
        
        with col1:
            refill_product = st.selectbox(
                "Product to Refill",
                options=stock_data['product_name'].tolist(),
                key="refill_product"
            )
        
        with col2:
            refill_qty = st.number_input("Quantity Received", min_value=1, value=10, key="refill_qty")
        
        with col3:
            # Auto-calculate cost based on existing unit cost
            product_info = stock_data[stock_data['product_name'] == refill_product].iloc[0]
            calculated_cost = refill_qty * product_info['unit_cost']
            st.metric("Total Cost ₹", f"₹{calculated_cost:.2f}", help="Auto-calculated: Quantity × Unit Cost")
        
        with col4:
            if st.button("➕ Add Refill", key="add_refill"):
                product_info = stock_data[stock_data['product_name'] == refill_product].iloc[0]
                total_cost = refill_qty * product_info['unit_cost']
                st.session_state.refill_entries.append({
                    'product_id': product_info['product_id'],
                    'product_name': refill_product,
                    'quantity': refill_qty,
                    'total_cost': total_cost,
                    'unit_cost': product_info['unit_cost'],  # Use existing unit cost
                    'current_stock': product_info['current_stock'],
                    'new_stock': product_info['current_stock'] + refill_qty
                })
                st.rerun()
        
        # Display pending refill entries
        if st.session_state.refill_entries:
            st.markdown("#### Pending Stock Refills")
            refill_df = pd.DataFrame(st.session_state.refill_entries)
            
            # Format the display
            display_df = refill_df.copy()
            display_df['Unit Cost'] = display_df['unit_cost'].apply(lambda x: f"₹{x:.2f}")
            display_df['Total Cost'] = display_df['total_cost'].apply(lambda x: f"₹{x:.2f}")
            display_df = display_df[['product_name', 'quantity', 'current_stock', 'new_stock', 'Unit Cost', 'Total Cost']]
            display_df.columns = ['Product', 'Qty Added', 'Current Stock', 'New Stock', 'Unit Cost', 'Total Cost']
            
            st.dataframe(display_df, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📦 Process All Refills", type="primary"):
                    success_count = 0
                    for entry in st.session_state.refill_entries:
                        if add_stock_refill(
                            entry['product_id'], 
                            entry['quantity'], 
                            entry['unit_cost']
                        ):
                            success_count += 1
                    
                    st.success(f"✅ Processed {success_count} stock refills!")
                    st.session_state.refill_entries = []
                    st.cache_data.clear()
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Clear All Refills"):
                    st.session_state.refill_entries = []
                    st.rerun()
            
            with col3:
                total_cost = sum(entry['total_cost'] for entry in st.session_state.refill_entries)
                st.metric("Total Investment", f"₹{total_cost:.2f}")
        
        # Stock status overview
        st.markdown("#### Current Stock Status")
        
        # Create stock status display
        stock_display = stock_data.copy()
        stock_display['Status'] = stock_display.apply(lambda row: 
            '🔴 Critical' if row['current_stock'] <= row['reorder_point'] * 0.5 
            else '🟡 Low' if row['current_stock'] <= row['reorder_point']
            else '🟢 Good', axis=1)
        
        stock_display['Value'] = (stock_display['current_stock'] * stock_display['unit_cost']).apply(lambda x: f"₹{x:.2f}")
        
        display_cols = ['product_name', 'current_stock', 'reorder_point', 'Status', 'Value']
        display_stock = stock_display[display_cols]
        display_stock.columns = ['Product', 'Current Stock', 'Reorder Point', 'Status', 'Value']
        
        st.dataframe(display_stock, use_container_width=True)

def add_sale_to_system(product_id: str, product_name: str, quantity: int, 
                      selling_price: float, customer_id: str) -> bool:
    """Add a sale to the system and update stock levels"""
    try:
        import time
        
        # Create new sale record (micro enterprises use retail channel by default)
        new_sale = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'product_id': product_id,
            'product_name': product_name,
            'quantity_sold': quantity,
            'unit_price': selling_price,
            'customer_id': customer_id,
            'sales_channel': 'retail',  # Default for micro enterprises
            'price': ''  # Keep empty for compatibility
        }
        
        # Load current sales data
        try:
            sales_data = pd.read_csv('data/sales.csv')
        except:
            sales_data = pd.DataFrame()
        
        # Add new sale
        new_sale_df = pd.DataFrame([new_sale])
        sales_data = pd.concat([sales_data, new_sale_df], ignore_index=True)
        
        # Save updated sales data
        sales_data.to_csv('data/sales.csv', index=False)
        
        # Update stock levels
        stock_data = pd.read_csv('data/stock.csv')
        
        # Find the product and reduce stock
        product_idx = stock_data[stock_data['product_id'] == product_id].index
        if len(product_idx) > 0:
            idx = product_idx[0]
            current_stock = stock_data.loc[idx, 'current_stock']
            new_stock = current_stock - quantity
            
            if new_stock >= 0:
                stock_data.loc[idx, 'current_stock'] = new_stock
                stock_data.loc[idx, 'last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Save updated stock data
                stock_data.to_csv('data/stock.csv', index=False)
                
                return True
            else:
                st.warning(f"Insufficient stock! Only {current_stock} units available.")
                return False
        else:
            st.error(f"Product {product_id} not found in stock data.")
            return False
            
    except Exception as e:
        st.error(f"Error adding sale: {e}")
        return False

def add_stock_refill(product_id: str, quantity: int, unit_cost: float) -> bool:
    """Add stock refill to the system and update stock levels"""
    try:
        # Load current stock data
        stock_data = pd.read_csv('data/stock.csv')
        
        # Find the product
        product_mask = stock_data['product_id'] == product_id
        
        if product_mask.any():
            idx = stock_data[product_mask].index[0]
            current_stock = stock_data.loc[idx, 'current_stock']
            new_stock = current_stock + quantity
            
            # Update stock data
            stock_data.loc[idx, 'current_stock'] = new_stock
            stock_data.loc[idx, 'unit_cost'] = unit_cost  # Update unit cost with latest purchase price
            stock_data.loc[idx, 'last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Save updated stock data
            stock_data.to_csv('data/stock.csv', index=False)
            
            return True
        else:
            st.error(f"Product {product_id} not found in stock data.")
            return False
            
    except Exception as e:
        st.error(f"Error adding stock refill: {e}")
        return False

def process_bulk_sales(bulk_sales_df: pd.DataFrame) -> int:
    """Process bulk sales from uploaded file"""
    success_count = 0
    
    for _, row in bulk_sales_df.iterrows():
        try:
            # Get product name from product_id
            stock_data = pd.read_csv('data/stock.csv')
            product_info = stock_data[stock_data['product_id'] == row['product_id']]
            
            if len(product_info) > 0:
                product_name = product_info.iloc[0]['product_name']
                
                if add_sale_to_system(
                    row['product_id'],
                    product_name,
                    int(row['quantity_sold']),
                    float(row['unit_price']),
                    row.get('customer_id', f"C{datetime.now().strftime('%Y%m%d%H%M%S')}")
                ):
                    success_count += 1
        except Exception as e:
            st.error(f"Error processing row: {e}")
            continue
    
    return success_count

def main():
    """Main dashboard application"""
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", [
        "📊 Overview",
        "🛒 Daily Sales Entry",
        "🤖 AI Planning",
        "📈 Analytics",
        "🔮 Forecasting"
    ])
    
    # System status
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")
    
    if st.session_state.agents_initialized:
        st.sidebar.success("✅ AI Agents Online")
    else:
        st.sidebar.warning("⚠️ AI Agents Not Initialized")
    
    if st.session_state.last_update:
        st.sidebar.info(f"Last Update: {st.session_state.last_update.strftime('%H:%M:%S')}")
    
    # Page routing
    if page == "📊 Overview":
        create_overview_dashboard()
    elif page == "🛒 Daily Sales Entry":
        create_daily_sales_entry()
    elif page == "🤖 AI Planning":
        create_ai_planning_dashboard()
    elif page == "📈 Analytics":
        create_analytics_dashboard()
    elif page == "🔮 Forecasting":
        create_demand_forecasting_dashboard()

if __name__ == "__main__":
    main()
