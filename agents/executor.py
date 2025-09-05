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
    supplier_id: str
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
            self.supplier_data = pd.read_csv('data/suppliers.csv')
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
            supplier_id="SINGLE_SUPPLIER",  # Single supplier for all orders
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
            # Send order email to supplier
            success = self.send_order_email(order_request, order_id)
            
            if success:
                # Add some random variation to cost
                cost_variation = np.random.uniform(0.98, 1.02)
                actual_cost = order_request.estimated_cost * cost_variation
                
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
    
    def send_order_email(self, order_request: OrderRequest, order_id: str) -> bool:
        """Send order email to supplier"""
        try:
            # Create email subject and body
            subject = f"Purchase Order {order_id} - {order_request.product_name}"
            
            body = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #4CAF50; color: white; padding: 15px; border-radius: 5px; }}
                    .content {{ background-color: #f9f9f9; padding: 20px; margin: 10px 0; border-radius: 5px; }}
                    .order-details {{ background-color: white; padding: 15px; margin: 10px 0; border-left: 4px solid #4CAF50; }}
                    .footer {{ color: #666; font-size: 12px; margin-top: 20px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h2>📦 PURCHASE ORDER</h2>
                </div>
                
                <div class="content">
                    <p>Dear Supplier,</p>
                    <p>We would like to place the following order based on our AI-predicted demand analysis:</p>
                    
                    <div class="order-details">
                        <h3>Order Details:</h3>
                        <ul>
                            <li><strong>Order ID:</strong> {order_id}</li>
                            <li><strong>Product ID:</strong> {order_request.product_id}</li>
                            <li><strong>Product Name:</strong> {order_request.product_name}</li>
                            <li><strong>Quantity:</strong> {order_request.quantity} units</li>
                            <li><strong>Estimated Cost:</strong> ₹{order_request.estimated_cost:,.2f}</li>
                            <li><strong>Priority:</strong> {order_request.priority}</li>
                            <li><strong>Requested Delivery:</strong> {order_request.expected_delivery.strftime('%Y-%m-%d')}</li>
                        </ul>
                    </div>
                    
                    <p><strong>Note:</strong> This order quantity has been calculated using AI demand prediction to optimize inventory levels based on forecasted customer demand.</p>
                    
                    <p>Please confirm receipt of this order and provide:</p>
                    <ul>
                        <li>Order confirmation</li>
                        <li>Final pricing</li>
                        <li>Expected delivery date</li>
                        <li>Tracking information when available</li>
                    </ul>
                    
                    <p>Thank you for your continued partnership.</p>
                    
                    <p>Best regards,<br>
                    Agentic Inventory Management System<br>
                    Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="footer">
                    <p>This is an automated order generated by AI analysis. Please contact us if you have any questions.</p>
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
            
            self.logger.info(f"Order email sent successfully for {order_request.product_name} to {self.supplier_email}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send order email: {e}")
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
            'supplier_id': order_request.supplier_id,
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
    
    async def execute_inventory_plans(self, plans: List[InventoryPlan]) -> Dict[str, Any]:
        """Execute multiple inventory plans"""
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
        
        # Execute urgent plans first
        self.logger.info(f"Executing {len(urgent_plans)} urgent inventory plans")
        
        for plan in urgent_plans:
            if plan.reorder_quantity > 0:
                order_request = self.create_order_request(plan)
                order_result = await self.execute_order(order_request)
                
                results['urgent_executed'] += 1
                if order_result.success:
                    results['successful_orders'] += 1
                    results['total_cost'] += order_result.actual_cost or order_request.estimated_cost
                else:
                    results['failed_orders'] += 1
                
                results['order_details'].append({
                    'product_name': plan.product_name,
                    'order_id': order_result.order_id,
                    'success': order_result.success,
                    'message': order_result.message,
                    'cost': order_result.actual_cost or order_request.estimated_cost
                })
        
        # Execute medium priority plans (limit to avoid overwhelming suppliers)
        medium_to_execute = medium_plans[:5]  # Limit to 5 medium priority orders
        
        self.logger.info(f"Executing {len(medium_to_execute)} medium priority inventory plans")
        
        for plan in medium_to_execute:
            if plan.reorder_quantity > 0:
                order_request = self.create_order_request(plan)
                order_result = await self.execute_order(order_request)
                
                results['medium_executed'] += 1
                if order_result.success:
                    results['successful_orders'] += 1
                    results['total_cost'] += order_result.actual_cost or order_request.estimated_cost
                else:
                    results['failed_orders'] += 1
                
                results['order_details'].append({
                    'product_name': plan.product_name,
                    'order_id': order_result.order_id,
                    'success': order_result.success,
                    'message': order_result.message,
                    'cost': order_result.actual_cost or order_request.estimated_cost
                })
        
        self.logger.info(
            f"Execution complete: {results['successful_orders']} successful, "
            f"{results['failed_orders']} failed, Total cost: ${results['total_cost']:.2f}"
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
    print(f"Total cost: ${results['total_cost']:.2f}")
    
    print("\nOrder Details:")
    for order in results['order_details']:
        print(f"- {order['product_name']}: {order['order_id']} - {'SUCCESS' if order['success'] else 'FAILED'}")
