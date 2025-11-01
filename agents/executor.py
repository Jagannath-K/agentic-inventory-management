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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
    
    # Email configuration from environment variables
    EMAIL_SMTP_SERVER = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
    EMAIL_SMTP_PORT = int(os.getenv('EMAIL_PORT', '587'))
    EMAIL_USER = os.getenv('EMAIL_USER')
    EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
    
    def __init__(self):
        super().__init__("ExecutorAgent")
        self.stock_data = None
        self.supplier_data = None
        self.pending_orders = []
        self.completed_orders = []
        self.order_counter = 1000
        self.demand_predictor = DemandPredictor()
        self.notification_system = NotificationSystem()
        self.supplier_email = os.getenv('SUPPLIER_EMAIL', 'supplier@example.com')  # From environment
        
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
        product_info, _ = self._find_product_by_id(plan.product_id)
        
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
        """Place order (always uses batch email mode)"""
        # Generate order ID
        order_id = f"ORD-{self.order_counter:06d}"
        self.order_counter += 1
        
        try:
            # Use fixed pricing - no random variations for consistency
            actual_cost = order_request.estimated_cost
            
            # Always use batch email mode
            message = "Order processed successfully (pending batch email)"
            self.logger.info(f"Order {order_id} processed for batch email for {order_request.product_name}")
            
            result = OrderResult(
                order_id=order_id,
                success=True,
                message=message,
                estimated_delivery=order_request.expected_delivery,
                actual_cost=actual_cost
            )
                
        except Exception as e:
            result = OrderResult(
                order_id=order_id,
                success=False,
                message=f"Order processing error: {str(e)}"
            )
            self.logger.error(f"Order {order_id} error: {e}")
        
        return result
    
    def _get_actual_cost(self, order_result: OrderResult, order_request: OrderRequest) -> float:
        """Helper method to get actual cost, falling back to estimated cost if needed"""
        return order_result.actual_cost or order_request.estimated_cost
    
    def _find_product_by_id(self, product_id: str) -> tuple[pd.Series, int]:
        """Helper method to find product info and index by product_id"""
        if self.stock_data is None:
            self.load_data()
        
        product_mask = self.stock_data['product_id'] == product_id
        product_idx = self.stock_data[product_mask].index[0]
        product_info = self.stock_data.iloc[product_idx]
        return product_info, product_idx
    
    def send_consolidated_order_email(self, order_requests: List[OrderRequest], order_ids: List[str], actual_costs: List[float] = None) -> bool:
        """Send consolidated order email with all orders in a single table"""
        try:
            if not order_requests:
                return True
                
            # Use actual costs if provided, otherwise use estimated costs
            if actual_costs is None:
                actual_costs = [req.estimated_cost for req in order_requests]
            
            # Store current datetime for consistency across email
            current_time = datetime.now()
            
            # Create email subject
            order_count = len(order_requests)
            total_value = sum(actual_costs)
            subject = f"📦 Purchase Order - {order_count} Items (₹{total_value:,.2f}) - {current_time.strftime('%Y-%m-%d')}"
            
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
                                <li><strong>Order Date:</strong> {current_time.strftime('%Y-%m-%d %H:%M:%S')}</li>
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
                        <p>🤖 This order was automatically generated by AI analysis • Generated on {current_time.strftime('%Y-%m-%d %H:%M:%S')} • Please contact us for any questions</p>
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
            smtp_server = self.EMAIL_SMTP_SERVER
            smtp_port = self.EMAIL_SMTP_PORT
            email_user = self.EMAIL_USER
            email_password = self.EMAIL_PASSWORD
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = email_user
            msg['To'] = self.supplier_email
            msg['Bcc'] = os.getenv('STOCK_ALERT_BCC', '')  # BCC copy from environment
            
            # Attach HTML body
            html_part = MIMEText(body, 'html')
            msg.attach(html_part)
            
            # Send email (BCC recipients are automatically included)
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(email_user, email_password)
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Consolidated order email sent successfully for {order_count} items to {self.supplier_email} (BCC: {os.getenv('STOCK_ALERT_BCC', 'none')})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send consolidated order email: {e}")
            return False
    
    def update_inventory_records(self, order_request: OrderRequest, order_result: OrderResult) -> bool:
        """Record order placement (NOT stock receipt) - track pending orders"""
        try:
            if not order_result.success:
                return False
            
            # Find the product in stock data
            _, product_idx = self._find_product_by_id(order_request.product_id)
            
            pending_order = {
                'order_id': order_result.order_id,
                'product_id': order_request.product_id,
                'product_name': order_request.product_name,
                'quantity_ordered': order_request.quantity,
                'cost': order_result.actual_cost or order_request.estimated_cost,
                'order_date': order_request.requested_date.isoformat(),
                'expected_delivery': order_request.expected_delivery.isoformat(),
                'status': 'PENDING_DELIVERY',
                'priority': order_request.priority
            }
            
            # Add to pending orders list (in memory for now)
            self.pending_orders.append(pending_order)
            
            # Update last_updated timestamp for tracking
            self.stock_data.at[product_idx, 'last_updated'] = datetime.now()
            
            # Save updated stock data (without changing current_stock yet)
            self.stock_data.to_csv('data/stock.csv', index=False)
            
            cost_display = self._get_actual_cost(order_result, order_request)
            self.logger.info(f"Recorded PENDING order: {order_request.quantity} units of {order_request.product_name} (Order ID: {order_result.order_id})")
            self.logger.info(f"⚠️  Stock will be updated only when delivery is confirmed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating inventory records: {e}")
            return False
    
    def confirm_delivery(self, order_id: str, received_quantity: Optional[int] = None) -> Dict[str, Any]:
        """Confirm delivery and update actual stock levels"""
        try:
            # Find the pending order
            pending_order = None
            order_index = None
            
            for i, order in enumerate(self.pending_orders):
                if order['order_id'] == order_id:
                    pending_order = order
                    order_index = i
                    break
            
            if not pending_order:
                return {
                    'success': False,
                    'message': f"Order {order_id} not found in pending orders"
                }
            
            if pending_order['status'] != 'PENDING_DELIVERY':
                return {
                    'success': False,
                    'message': f"Order {order_id} is not pending delivery (status: {pending_order['status']})"
                }
            
            # Use received quantity or default to ordered quantity
            if received_quantity is None:
                received_quantity = pending_order['quantity_ordered']
            
            # Load current stock data
            if self.stock_data is None:
                self.load_data()
            
            # Find and update the product stock
            _, product_idx = self._find_product_by_id(pending_order['product_id'])
            current_stock = self.stock_data.at[product_idx, 'current_stock']
            new_stock = current_stock + received_quantity
            
            # Update stock levels
            self.stock_data.at[product_idx, 'current_stock'] = new_stock
            self.stock_data.at[product_idx, 'last_updated'] = datetime.now()
            
            # Save updated stock data
            self.stock_data.to_csv('data/stock.csv', index=False)
            
            # Update order status
            self.pending_orders[order_index]['status'] = 'DELIVERED'
            self.pending_orders[order_index]['delivery_date'] = datetime.now().isoformat()
            self.pending_orders[order_index]['received_quantity'] = received_quantity
            
            # Move to completed orders
            completed_order = self.pending_orders.pop(order_index)
            self.completed_orders.append(completed_order)
            
            self.logger.info(f"✅ DELIVERY CONFIRMED: {received_quantity} units of {pending_order['product_name']}")
            self.logger.info(f"📦 Stock updated: {current_stock} → {new_stock} units")
            
            return {
                'success': True,
                'message': f"Delivery confirmed for {pending_order['product_name']}",
                'product_name': pending_order['product_name'],
                'received_quantity': received_quantity,
                'old_stock': current_stock,
                'new_stock': new_stock,
                'order_id': order_id
            }
            
        except Exception as e:
            self.logger.error(f"Error confirming delivery for order {order_id}: {e}")
            return {
                'success': False,
                'message': f"Error confirming delivery: {str(e)}"
            }
    
    def get_pending_orders(self) -> List[Dict[str, Any]]:
        """Get list of orders pending delivery"""
        return [order for order in self.pending_orders if order['status'] == 'PENDING_DELIVERY']
    
    def _find_order_in_list(self, order_id: str, order_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Helper method to find an order by ID in a given list"""
        for order in order_list:
            if order['order_id'] == order_id:
                return order
        return None
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get status of a specific order"""
        # Search in pending orders first, then completed orders
        for order_list in [self.pending_orders, self.completed_orders]:
            found_order = self._find_order_in_list(order_id, order_list)
            if found_order:
                return {
                    'found': True,
                    'status': found_order['status'],
                    'order_details': found_order
                }
        
        return {
            'found': False,
            'message': f"Order {order_id} not found"
        }
    
    async def execute_order(self, order_request: OrderRequest) -> OrderResult:
        """Execute a single order request (always uses batch email mode)"""
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
        
        # Place order
        order_result = await self.simulate_order_placement(order_request)
        
        # Update records and log transaction
        if order_result.success:
            self.update_inventory_records(order_request, order_result)
        
        return order_result
    
    async def _process_plan_batch(self, plans: List[InventoryPlan], plan_type: str, results: Dict[str, Any], 
                                 all_order_requests: List, all_order_ids: List, all_actual_costs: List) -> None:
        """Helper method to process a batch of plans """
        self.logger.info(f"Processing {len(plans)} {plan_type} priority inventory plans")
        
        for plan in plans:
            if plan.reorder_quantity > 0:
                order_request = self.create_order_request(plan)
                order_result = await self.execute_order(order_request)
                
                # Update counters based on plan type
                if plan_type == "urgent":
                    results['urgent_executed'] += 1
                else:  # medium
                    results['medium_executed'] += 1
                
                if order_result.success:
                    results['successful_orders'] += 1
                    actual_cost = self._get_actual_cost(order_result, order_request)
                    results['total_cost'] += actual_cost
                    
                    # Collect successful orders for consolidated email
                    all_order_requests.append(order_request)
                    all_order_ids.append(order_result.order_id)
                    all_actual_costs.append(actual_cost)
                else:
                    results['failed_orders'] += 1
                
                results['order_details'].append({
                    'product_name': plan.product_name,
                    'order_id': order_result.order_id,
                    'success': order_result.success,
                    'message': order_result.message,
                    'cost': self._get_actual_cost(order_result, order_request)
                })

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
        
        # Collect all successful orders for consolidated email
        all_order_requests = []
        all_order_ids = []
        all_actual_costs = []
        
        # Process urgent plans first
        await self._process_plan_batch(urgent_plans, "urgent", results, 
                                     all_order_requests, all_order_ids, all_actual_costs)
        
        medium_to_execute = medium_plans[:10]  # Limit to 10 medium priority orders
        skipped_medium = medium_plans[10:] if len(medium_plans) > 10 else []
        
        if skipped_medium:
            skipped_names = [plan.product_name for plan in skipped_medium]
            self.logger.info(f"Skipped {len(skipped_medium)} medium priority items: {', '.join(skipped_names)}")
        
        await self._process_plan_batch(medium_to_execute, "medium", results, 
                                     all_order_requests, all_order_ids, all_actual_costs)
        
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
        elif action == "confirm_delivery":
            order_id = kwargs.get('order_id')
            received_quantity = kwargs.get('received_quantity')
            return self.confirm_delivery(order_id, received_quantity)
        elif action == "get_pending_orders":
            return self.get_pending_orders()
        elif action == "get_order_status":
            order_id = kwargs.get('order_id')
            return self.get_order_status(order_id)
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
