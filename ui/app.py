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
        suppliers_data = pd.read_csv('data/suppliers.csv')
        
        # Parse dates with modern pandas approach
        sales_data['date'] = pd.to_datetime(sales_data['date'], errors='coerce')
        stock_data['last_updated'] = pd.to_datetime(stock_data['last_updated'], errors='coerce')
        
        return sales_data, stock_data, suppliers_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

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
    sales_data, stock_data, suppliers_data = load_data()
    
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
    
    # Display plans if available
    if 'current_plans' in st.session_state and st.session_state.current_plans:
        plans = st.session_state.current_plans
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            critical_count = len([p for p in plans if p.urgency_level == 'CRITICAL'])
            st.metric("Critical Items", critical_count)
        
        with col2:
            high_count = len([p for p in plans if p.urgency_level == 'HIGH'])
            st.metric("High Priority", high_count)
        
        with col3:
            total_reorder_value = sum(p.reorder_quantity * 
                                    st.session_state.planner.stock_data[
                                        st.session_state.planner.stock_data['product_id'] == p.product_id
                                    ]['unit_cost'].iloc[0] for p in plans if p.reorder_quantity > 0)
            st.metric("Total Reorder Value", f"₹{total_reorder_value:,.2f}")
        
        with col4:
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
            plan_data.append({
                'Product': plan.product_name,
                'Current Stock': plan.current_stock,
                'Predicted Demand (30d)': f"{plan.predicted_demand:.1f}",
                'Reorder Quantity': plan.reorder_quantity,
                'Urgency': plan.urgency_level,
                'Reorder Date': plan.reorder_date.strftime('%Y-%m-%d'),
                'Confidence': f"{plan.confidence_score:.1%}",
                'Reasoning': plan.reasoning[:100] + "..." if len(plan.reasoning) > 100 else plan.reasoning
            })
        
        if plan_data:
            df = pd.DataFrame(plan_data)
            
            # Color code urgency
            def highlight_urgency(row):
                if row['Urgency'] == 'CRITICAL':
                    return ['background-color: #ffebee'] * len(row)
                elif row['Urgency'] == 'HIGH':
                    return ['background-color: #fff3e0'] * len(row)
                else:
                    return [''] * len(row)
            
            st.dataframe(df.style.apply(highlight_urgency, axis=1), use_container_width=True)
            
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
        if st.button("Generate Report", type="primary"):
            with st.spinner("AI agent is analyzing system performance..."):
                try:
                    report = st.session_state.reflector.create_optimization_report()
                    st.session_state.analytics_report = report
                    st.success("Analytics report generated!")
                except Exception as e:
                    st.error(f"Error generating report: {e}")
    
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
        
        col1, col2, col3, col4 = st.columns(4)
        
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
        
        with col4:
            st.metric("Suppliers Evaluated", report['summary']['suppliers_evaluated'])
        
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
    sales_data, stock_data, suppliers_data = load_data()
    
    if stock_data is None:
        return
    
    # Product selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_product = st.selectbox("Select Product for Forecasting", 
                                      stock_data['product_id'].tolist())
    
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
            # Forecast chart
            st.subheader(f"Demand Forecast for {selected_product}")
            
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
                
                fig.update_layout(title=f"Sales History and Forecast for {selected_product}",
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
    st.markdown('<h1 class="main-header">🛒 Daily Sales Entry</h1>', unsafe_allow_html=True)
    st.markdown("**Enter your daily sales to keep inventory updated and improve AI predictions**")
    
    # Load current data
    sales_data, stock_data, suppliers_data = load_data()
    
    if stock_data is None:
        st.error("Unable to load stock data. Please check data files.")
        return
    
    # Create tabs for different entry methods
    tab1, tab2 = st.tabs([" Sales Entry", "📊 Today's Summary"])
    
    with tab1:
        st.subheader("Sales Entry")
        
        # Initialize session state for bulk entries
        if 'bulk_entries' not in st.session_state:
            st.session_state.bulk_entries = []
        
        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
        
        with col1:
            bulk_product = st.selectbox(
                "Product",
                options=stock_data['product_name'].tolist(),
                key="bulk_product"
            )
        
        with col2:
            bulk_qty = st.number_input("Qty", min_value=1, value=1, key="bulk_qty")
        
        with col3:
            bulk_price = st.number_input("Price ₹", min_value=0.0, value=100.0, key="bulk_price")
        
        with col4:
            bulk_channel = st.selectbox("Channel", ["retail", "wholesale", "online"], key="bulk_channel")
        
        with col5:
            if st.button("➕ Add", key="add_bulk"):
                product_info = stock_data[stock_data['product_name'] == bulk_product].iloc[0]
                st.session_state.bulk_entries.append({
                    'product_id': product_info['product_id'],
                    'product_name': bulk_product,
                    'quantity': bulk_qty,
                    'price': bulk_price,
                    'channel': bulk_channel,
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
                            f"C{datetime.now().strftime('%Y%m%d%H%M%S')}", 
                            entry['channel']
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
                    st.metric("Total Revenue", f"₹{today_sales['unit_price'].sum():.2f}")
                
                with col3:
                    st.metric("Items Sold", today_sales['quantity_sold'].sum())
                
                with col4:
                    st.metric("Avg Transaction", f"₹{today_sales['unit_price'].mean():.2f}")
                
                # Today's sales by product
                st.markdown("#### Today's Sales by Product")
                product_sales = today_sales.groupby('product_name').agg({
                    'quantity_sold': 'sum',
                    'unit_price': 'sum'
                }).round(2)
                
                if len(product_sales) > 0:
                    product_sales['Revenue'] = product_sales['unit_price']
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
                        y='unit_price',
                        title="Today's Revenue by Product",
                        labels={'unit_price': 'Revenue (₹)', 'product_name': 'Product'}
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.info("No sales recorded for today yet. Start adding sales using the Quick Entry tab!")
        else:
            st.info("No sales data available. Start recording your first sale!")

def add_sale_to_system(product_id: str, product_name: str, quantity: int, 
                      selling_price: float, customer_id: str, sales_channel: str) -> bool:
    """Add a sale to the system and update stock levels"""
    try:
        import time
        
        # Create new sale record
        new_sale = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'product_id': product_id,
            'product_name': product_name,
            'quantity_sold': quantity,
            'unit_price': selling_price,
            'customer_id': customer_id,
            'sales_channel': sales_channel,
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
                    row.get('customer_id', f"C{datetime.now().strftime('%Y%m%d%H%M%S')}"),
                    row.get('sales_channel', 'retail')
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
