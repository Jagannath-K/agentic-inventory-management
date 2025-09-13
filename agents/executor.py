"""
Executor Agent - Handles order execution and inventory updates
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import json
import asyncio
import os
import sys

# Add models to path for demand prediction
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .planner import BaseAgent, InventoryPlan
from models.predictor import DemandPredictor
from models.notification_system import NotificationSystem

@dataclass
class OrderRequest:
    """Data class for order requests"""
    product_id: str
    product_name: str
    quantity: int
    estimated_cost: float
    priority: str
    requested_date: datetime
    expected_delivery: datetime
    order_id: Optional[str] = None
    status: str = "PENDING"

@dataclass
class OrderResult:
    """Data class for order execution results"""
    order_id: str
    success: bool
    message: str
    estimated_delivery: Optional[datetime] = None
    actual_cost: Optional[float] = None

class ExecutorAgent(BaseAgent):
    """
    AI Agent responsible for executing inventory decisions and managing orders
    """
    
    def __init__(self):
        super().__init__("ExecutorAgent")
        self.stock_data = None
        self.supplier_data = None
        self.pending_orders = []
        self.completed_orders = []
        self.order_counter = 1000
        self.demand_predictor = DemandPredictor()
        self.notification_system = NotificationSystem()
        self.supplier_email = "jagannath.backup.2005@gmail.com"  # Single supplier email
        
    def load_data(self) -> None:
        """Load necessary data for execution"""
        try:
            self.stock_data = pd.read_csv('data/stock.csv')
            # Removed supplier data loading - using single supplier for all items
            # Parse dates with error handling for mixed formats
            self.stock_data['last_updated'] = pd.to_datetime(self.stock_data['last_updated'], errors='coerce')
            self.logger.info("Executor data loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def create_order_request(self, plan: InventoryPlan) -> OrderRequest:
        """Create an order request from an inventory plan with AI-predicted demand for grocery items"""
        if self.stock_data is None:
            self.load_data()
        
        # Get product and supplier information
        product_info = self.stock_data[self.stock_data['product_id'] == plan.product_id].iloc[0]
        
        # Calculate order quantity based on predicted demand (30-day forecast)
        try:
            # Get predicted demand for the next 30 days
            future_date = datetime.now() + timedelta(days=30)
            demand_result = self.demand_predictor.predict_demand(plan.product_id, future_date)
            predicted_monthly_demand = demand_result.get('predicted_demand', 0)
            prediction_reasoning = demand_result.get('reasoning', 'No reasoning available')
            
            # Calculate safety stock (20% of predicted demand)
            safety_stock = max(int(predicted_monthly_demand * 0.2), plan.reorder_quantity)
            
            # Order quantity = predicted demand + safety stock - current stock
            current_stock = product_info['current_stock']
            order_quantity = max(int(predicted_monthly_demand + safety_stock - current_stock), plan.reorder_quantity)
            
            self.logger.info(f"Predicted demand for {plan.product_name}: {predicted_monthly_demand:.1f} units (30 days)")
            self.logger.info(f"AI Reasoning: {prediction_reasoning}")
            self.logger.info(f"Order quantity adjusted to: {order_quantity} units (includes safety stock)")
            
        except Exception as e:
            # Fallback to original plan quantity if prediction fails
            order_quantity = plan.reorder_quantity
            self.logger.warning(f"Could not predict demand for {plan.product_name}, using plan quantity: {e}")
        
        # Calculate estimated cost
        estimated_cost = order_quantity * product_info['unit_cost']
        
        # Calculate expected delivery date (assume 3 days for single supplier)
        expected_delivery = datetime.now() + timedelta(days=3)
        
        order_request = OrderRequest(
            product_id=plan.product_id,
            product_name=plan.product_name,
            quantity=order_quantity,  # Use predicted demand-based quantity
            estimated_cost=estimated_cost,
            priority=plan.urgency_level,
            requested_date=datetime.now(),
            expected_delivery=expected_delivery
        )
        
        return order_request
    
    def validate_order_request(self, order_request: OrderRequest) -> Dict[str, Any]:
        """Validate order request before execution"""
        validation_result = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Basic validation
        if order_request.quantity <= 0:
            validation_result['issues'].append("Order quantity must be greater than 0")
            validation_result['valid'] = False
        
        if order_request.estimated_cost <= 0:
            validation_result['issues'].append("Order cost must be greater than 0")
            validation_result['valid'] = False
        
        # Check if quantity is reasonable
        if self.stock_data is not None:
            product_info = self.stock_data[self.stock_data['product_id'] == order_request.product_id]
            if not product_info.empty:
                product = product_info.iloc[0]
                if order_request.quantity > product['max_stock'] * 2:  # Allow up to 2x max stock for predicted demand
                    validation_result['warnings'].append(
                        f"Order quantity ({order_request.quantity}) is very large compared to max stock ({product['max_stock']})"
                    )
        
        # Check email configuration
        if not self.supplier_email:
            validation_result['issues'].append("Supplier email not configured")
            validation_result['valid'] = False
        
        return validation_result
    
    async def simulate_order_placement(self, order_request: OrderRequest) -> OrderResult:
        """Place order and send email to supplier"""
        # Generate order ID
        order_id = f"ORD-{self.order_counter:06d}"
        self.order_counter += 1
        
        try:
            # Use fixed pricing - no random variations for consistency
            actual_cost = order_request.estimated_cost
            
            # Send order email to supplier
            success = self.send_order_email(order_request, order_id, actual_cost)
            
            if success:
                result = OrderResult(
                    order_id=order_id,
                    success=True,
                    message=f"Order email sent successfully to supplier",
                    estimated_delivery=order_request.expected_delivery,
                    actual_cost=actual_cost
                )
                
                self.logger.info(f"Order {order_id} sent successfully for {order_request.product_name}")
            else:
                result = OrderResult(
                    order_id=order_id,
                    success=False,
                    message="Failed to send order email to supplier"
                )
                
                self.logger.error(f"Order {order_id} failed for {order_request.product_name}")
                
        except Exception as e:
            result = OrderResult(
                order_id=order_id,
                success=False,
                message=f"Order processing error: {str(e)}"
            )
            self.logger.error(f"Order {order_id} error: {e}")
        
        return result
    
    async def simulate_order_placement_without_email(self, order_request: OrderRequest) -> OrderResult:
        """Place order without sending individual email (for batch processing)"""
        # Generate order ID
        order_id = f"ORD-{self.order_counter:06d}"
        self.order_counter += 1
        
        try:
            # Use fixed pricing - no random variations for consistency
            actual_cost = order_request.estimated_cost
            
            result = OrderResult(
                order_id=order_id,
                success=True,
                message=f"Order processed successfully (pending batch email)",
                estimated_delivery=order_request.expected_delivery,
                actual_cost=actual_cost
            )
            
            self.logger.info(f"Order {order_id} processed for batch email for {order_request.product_name}")
                
        except Exception as e:
            result = OrderResult(
                order_id=order_id,
                success=False,
                message=f"Order processing error: {str(e)}"
            )
            self.logger.error(f"Order {order_id} error: {e}")
        
        return result
    
    def send_order_email(self, order_request: OrderRequest, order_id: str, actual_cost: float = None) -> bool:
        """Send order email to supplier - individual order (deprecated, use send_consolidated_order_email)"""
        # This method is kept for compatibility but consolidated email is preferred
        if actual_cost is None:
            actual_cost = order_request.estimated_cost
        return self.send_consolidated_order_email([order_request], [order_id], [actual_cost])
    
    def send_consolidated_order_email(self, order_requests: List[OrderRequest], order_ids: List[str], actual_costs: List[float] = None) -> bool:
        """Send consolidated order email with all orders in a single table"""
        try:
            if not order_requests:
                return True
                
            # Use actual costs if provided, otherwise use estimated costs
            if actual_costs is None:
                actual_costs = [req.estimated_cost for req in order_requests]
                
            # Create email subject
            order_count = len(order_requests)
            total_value = sum(actual_costs)
            subject = f"📦 Purchase Order - {order_count} Items (₹{total_value:,.2f}) - {datetime.now().strftime('%Y-%m-%d')}"
            
            # Create consolidated order table
            order_rows = ""
            for i, (order_request, order_id, actual_cost) in enumerate(zip(order_requests, order_ids, actual_costs), 1):
                order_rows += f"""
                <tr style="border-bottom: 1px solid #ddd;">
                    <td style="padding: 12px; text-align: center; font-weight: bold;">{i}</td>
                    <td style="padding: 12px; text-align: center; font-family: monospace; background-color: #f0f0f0;">{order_id}</td>
                    <td style="padding: 12px; text-align: center; font-family: monospace; color: #666;">{order_request.product_id}</td>
                    <td style="padding: 12px; font-weight: bold;">{order_request.product_name}</td>
                    <td style="padding: 12px; text-align: center; font-size: 16px; font-weight: bold;">{order_request.quantity}</td>
                    <td style="padding: 12px; text-align: right; font-weight: bold; color: #2e7d32;">₹{actual_cost:,.2f}</td>
                    <td style="padding: 12px; text-align: center;">
                        <span style="padding: 4px 8px; border-radius: 12px; font-size: 11px; font-weight: bold; 
                        {'background-color: #ffcdd2; color: #b71c1c;' if order_request.priority == 'CRITICAL' 
                         else 'background-color: #fff3e0; color: #e65100;' if order_request.priority == 'HIGH'
                         else 'background-color: #fff8e1; color: #f57c00;'}">{order_request.priority}</span>
                    </td>
                    <td style="padding: 12px; text-align: center;">{order_request.expected_delivery.strftime('%Y-%m-%d')}</td>
                </tr>
                """
            
            body = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                    .container {{ max-width: 1000px; margin: 0 auto; background-color: white; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                    .header {{ background: linear-gradient(135deg, #1f77b4, #4CAF50); color: white; padding: 25px; text-align: center; }}
                    .content {{ padding: 30px; }}
                    .summary {{ background-color: #e3f2fd; padding: 20px; margin: 20px 0; border-radius: 8px; border-left: 5px solid #1f77b4; }}
                    .order-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                    .order-table th {{ background-color: #1f77b4; color: white; padding: 15px 12px; font-weight: bold; text-align: center; }}
                    .order-table td {{ border-bottom: 1px solid #ddd; }}
                    .total-row {{ background-color: #f8f9fa; font-weight: bold; font-size: 16px; }}
                    .footer {{ background-color: #f8f9fa; padding: 20px; text-align: center; color: #666; font-size: 12px; }}
                    .ai-note {{ background-color: #fff3e0; padding: 15px; margin: 15px 0; border-radius: 6px; border-left: 4px solid #ff9800; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🤖 AI-Generated Purchase Order</h1>
                        <p style="margin: 5px 0; font-size: 18px;">Micro Enterprise Inventory Management</p>
                    </div>
                    
                    <div class="content">
                        <p style="font-size: 16px; color: #333;"><strong>Dear Supplier,</strong></p>
                        <p>We would like to place the following consolidated order based on our AI demand prediction analysis:</p>
                        
                        <div class="summary">
                            <h3 style="margin-top: 0; color: #1f77b4;">📊 Order Summary</h3>
                            <ul style="margin: 10px 0; padding-left: 20px;">
                                <li><strong>Total Items:</strong> {order_count} products</li>
                                <li><strong>Total Order Value:</strong> ₹{total_value:,.2f}</li>
                                <li><strong>Order Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
                                <li><strong>Request Type:</strong> AI-Optimized Inventory Replenishment</li>
                            </ul>
                        </div>
                        
                        <h3 style="color: #1f77b4;">📋 Detailed Order List</h3>
                        <table class="order-table">
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Order ID</th>
                                    <th>Product ID</th>
                                    <th>Product Name</th>
                                    <th>Quantity</th>
                                    <th>Estimated Cost</th>
                                    <th>Priority</th>
                                    <th>Required By</th>
                                </tr>
                            </thead>
                            <tbody>
                                {order_rows}
                                <tr class="total-row">
                                    <td colspan="5" style="padding: 15px; text-align: right; background-color: #f8f9fa;">
                                        <strong>TOTAL ORDER VALUE:</strong>
                                    </td>
                                    <td style="padding: 15px; text-align: right; background-color: #f8f9fa; color: #1f77b4; font-size: 18px;">
                                        <strong>₹{total_value:,.2f}</strong>
                                    </td>
                                    <td colspan="2" style="background-color: #f8f9fa;"></td>
                                </tr>
                            </tbody>
                        </table>
                        
                        <div class="ai-note">
                            <p style="margin: 0;"><strong>🤖 AI Analysis Note:</strong> These quantities have been calculated using machine learning algorithms that analyze historical sales patterns, seasonal trends, and current stock levels to optimize inventory for micro enterprise operations.</p>
                        </div>
                        
                        <h3 style="color: #1f77b4;">📞 Please Confirm:</h3>
                        <ul style="line-height: 1.6;">
                            <li>✅ Order confirmation and acceptance</li>
                            <li>💰 Final pricing (if different from estimates)</li>
                            <li>📅 Confirmed delivery dates for each item</li>
                            <li>🚚 Delivery method and tracking information</li>
                            <li>💳 Payment terms and preferred method</li>
                        </ul>
                        
                        <p style="margin: 25px 0; font-size: 16px;">Thank you for your continued partnership in supporting our micro enterprise operations.</p>
                        
                        <p><strong>Best regards,</strong><br>
                        <span style="color: #1f77b4; font-weight: bold;">Agentic AI Inventory Management System</span><br>
                        <small>Micro Enterprise Solutions</small></p>
                    </div>
                    
                    <div class="footer">
                        <p>🤖 This order was automatically generated by AI analysis • Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • Please contact us for any questions</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Send email using notification system
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            import smtplib
            import os
            
            # Email configuration
            smtp_server = "smtp.gmail.com"
            smtp_port = 587
            email_user = "kjagannath321@gmail.com"
            email_password = "hkpu bisz volr vjgr"
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = email_user
            msg['To'] = self.supplier_email
            
            # Attach HTML body
            html_part = MIMEText(body, 'html')
            msg.attach(html_part)
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(email_user, email_password)
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Consolidated order email sent successfully for {order_count} items to {self.supplier_email}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send consolidated order email: {e}")
            return False
        
        return result
    
    def update_inventory_records(self, order_request: OrderRequest, order_result: OrderResult) -> bool:
        """Update inventory records after successful order placement"""
        try:
            if not order_result.success:
                return False
            
            # Find the product in stock data
            product_idx = self.stock_data[self.stock_data['product_id'] == order_request.product_id].index[0]
            
            # Update expected stock (incoming inventory)
            # Note: In a real system, this would be tracked separately as "on order" inventory
            self.logger.info(f"Recorded incoming inventory: {order_request.quantity} units of {order_request.product_name}")
            
            # Update last_updated timestamp
            self.stock_data.at[product_idx, 'last_updated'] = datetime.now()
            
            # Save updated stock data
            self.stock_data.to_csv('data/stock.csv', index=False)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating inventory records: {e}")
            return False
    
    def log_order_transaction(self, order_request: OrderRequest, order_result: OrderResult) -> None:
        """Log order transaction for audit trail"""
        transaction = {
            'timestamp': datetime.now().isoformat(),
            'order_id': order_result.order_id,
            'product_id': order_request.product_id,
            'product_name': order_request.product_name,
            'quantity': order_request.quantity,
            'estimated_cost': order_request.estimated_cost,
            'actual_cost': order_result.actual_cost,
            'priority': order_request.priority,
            'success': order_result.success,
            'message': order_result.message,
            'expected_delivery': order_request.expected_delivery.isoformat() if order_request.expected_delivery else None,
            'estimated_delivery': order_result.estimated_delivery.isoformat() if order_result.estimated_delivery else None
        }
        
        # In a real system, this would be saved to a database
        # For now, we'll append to a JSON file
        try:
            import os
            log_file = 'data/order_log.json'
            
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(transaction)
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error logging transaction: {e}")
    
    async def execute_order(self, order_request: OrderRequest) -> OrderResult:
        """Execute a single order request"""
        self.logger.info(f"Executing order for {order_request.product_name} - Quantity: {order_request.quantity}")
        
        # Validate order
        validation = self.validate_order_request(order_request)
        
        if not validation['valid']:
            return OrderResult(
                order_id="VALIDATION_FAILED",
                success=False,
                message=f"Validation failed: {'; '.join(validation['issues'])}"
            )
        
        # Log warnings
        for warning in validation['warnings']:
            self.logger.warning(warning)
        
        # Place order
        order_result = await self.simulate_order_placement(order_request)
        
        # Update records and log transaction
        if order_result.success:
            self.update_inventory_records(order_request, order_result)
        
        self.log_order_transaction(order_request, order_result)
        
        return order_result
    
    async def execute_order_without_email(self, order_request: OrderRequest) -> OrderResult:
        """Execute a single order request without sending individual email (for batch processing)"""
        self.logger.info(f"Processing order for {order_request.product_name} - Quantity: {order_request.quantity} (batch mode)")
        
        # Validate order
        validation = self.validate_order_request(order_request)
        
        if not validation['valid']:
            return OrderResult(
                order_id="VALIDATION_FAILED",
                success=False,
                message=f"Validation failed: {'; '.join(validation['issues'])}"
            )
        
        # Log warnings
        for warning in validation['warnings']:
            self.logger.warning(warning)
        
        # Place order (without email sending)
        order_result = await self.simulate_order_placement_without_email(order_request)
        
        # Update records and log transaction
        if order_result.success:
            self.update_inventory_records(order_request, order_result)
        
        self.log_order_transaction(order_request, order_result)
        
        return order_result
    
    async def execute_inventory_plans(self, plans: List[InventoryPlan]) -> Dict[str, Any]:
        """Execute multiple inventory plans with consolidated email approach"""
        if self.stock_data is None:
            self.load_data()
        
        # Filter plans that need immediate action
        urgent_plans = [plan for plan in plans if plan.urgency_level in ['CRITICAL', 'HIGH']]
        medium_plans = [plan for plan in plans if plan.urgency_level == 'MEDIUM']
        
        results = {
            'total_plans': len(plans),
            'urgent_executed': 0,
            'medium_executed': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'total_cost': 0.0,
            'order_details': []
        }
        
        # Collect all orders to be executed (without sending individual emails)
        all_order_requests = []
        all_order_ids = []
        all_order_results = []
        all_actual_costs = []
        
        # Process urgent plans first
        self.logger.info(f"Processing {len(urgent_plans)} urgent inventory plans")
        
        for plan in urgent_plans:
            if plan.reorder_quantity > 0:
                order_request = self.create_order_request(plan)
                order_result = await self.execute_order_without_email(order_request)
                
                results['urgent_executed'] += 1
                if order_result.success:
                    results['successful_orders'] += 1
                    actual_cost = order_result.actual_cost or order_request.estimated_cost
                    results['total_cost'] += actual_cost
                    # Collect successful orders for consolidated email
                    all_order_requests.append(order_request)
                    all_order_ids.append(order_result.order_id)
                    all_order_results.append(order_result)
                    all_actual_costs.append(actual_cost)
                else:
                    results['failed_orders'] += 1
                
                results['order_details'].append({
                    'product_name': plan.product_name,
                    'order_id': order_result.order_id,
                    'success': order_result.success,
                    'message': order_result.message,
                    'cost': order_result.actual_cost or order_request.estimated_cost
                })
        
        # Process medium priority plans (increased limit since we use consolidated emails)
        medium_to_execute = medium_plans[:10]  # Increased limit to 10 medium priority orders
        skipped_medium = medium_plans[10:] if len(medium_plans) > 10 else []
        
        self.logger.info(f"Processing {len(medium_to_execute)} out of {len(medium_plans)} medium priority inventory plans")
        if skipped_medium:
            skipped_names = [plan.product_name for plan in skipped_medium]
            self.logger.info(f"Skipped medium priority items: {', '.join(skipped_names)}")
        
        for plan in medium_to_execute:
            if plan.reorder_quantity > 0:
                order_request = self.create_order_request(plan)
                order_result = await self.execute_order_without_email(order_request)
                
                results['medium_executed'] += 1
                if order_result.success:
                    results['successful_orders'] += 1
                    actual_cost = order_result.actual_cost or order_request.estimated_cost
                    results['total_cost'] += actual_cost
                    # Collect successful orders for consolidated email
                    all_order_requests.append(order_request)
                    all_order_ids.append(order_result.order_id)
                    all_order_results.append(order_result)
                    all_actual_costs.append(actual_cost)
                else:
                    results['failed_orders'] += 1
                
                results['order_details'].append({
                    'product_name': plan.product_name,
                    'order_id': order_result.order_id,
                    'success': order_result.success,
                    'message': order_result.message,
                    'cost': order_result.actual_cost or order_request.estimated_cost
                })
        
        # Send consolidated email with all successful orders
        if all_order_requests:
            self.logger.info(f"Sending consolidated email for {len(all_order_requests)} orders")
            email_success = self.send_consolidated_order_email(all_order_requests, all_order_ids, all_actual_costs)
            
            if email_success:
                results['consolidated_email_sent'] = True
                results['email_message'] = f"Consolidated email sent successfully for {len(all_order_requests)} orders"
            else:
                results['consolidated_email_sent'] = False
                results['email_message'] = "Failed to send consolidated email"
        else:
            results['consolidated_email_sent'] = False
            results['email_message'] = "No orders to send"
        
        self.logger.info(
            f"Execution complete: {results['successful_orders']} successful, "
            f"{results['failed_orders']} failed, Total cost: ₹{results['total_cost']:.2f}"
        )
        
        return results
    
    def process(self, action: str, **kwargs) -> Any:
        """Main processing method for the executor agent"""
        if action == "execute_plans":
            plans = kwargs.get('plans', [])
            return asyncio.run(self.execute_inventory_plans(plans))
        elif action == "execute_single_order":
            order_request = kwargs.get('order_request')
            return asyncio.run(self.execute_order(order_request))
        elif action == "validate_order":
            order_request = kwargs.get('order_request')
            return self.validate_order_request(order_request)
        else:
            raise ValueError(f"Unknown action: {action}")

if __name__ == "__main__":
    # Test the executor agent
    from .planner import PlannerAgent
    
    # Create some test plans
    planner = PlannerAgent()
    plans = planner.create_inventory_plan()
    
    # Execute the plans
    executor = ExecutorAgent()
    results = executor.process("execute_plans", plans=plans[:3])  # Execute first 3 plans
    
    print("\nExecution Results:")
    print(f"Successful orders: {results['successful_orders']}")
    print(f"Failed orders: {results['failed_orders']}")
    print(f"Total cost: ₹{results['total_cost']:.2f}")
    
    print("\nOrder Details:")
    for order in results['order_details']:
        print(f"- {order['product_name']}: {order['order_id']} - {'SUCCESS' if order['success'] else 'FAILED'}")
