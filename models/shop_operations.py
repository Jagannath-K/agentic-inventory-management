"""
Shop Operations Module
Handles daily transactions, stock updates, and inventory operations
"""

import pandas as pd
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
from .notification_system import NotificationSystem

logger = logging.getLogger(__name__)

class ShopOperations:
    """
    Handles day-to-day shop operations including sales and stock management
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.stock_file = 'data/stock.csv'
        self.sales_file = 'data/sales.csv'
        self.transactions_log = 'data/daily_transactions.json'
        
        # Initialize notification system
        self.notification_system = NotificationSystem()
        
        # Store config for notification settings
        self.config = config or {}
        
        # Default notification settings if not provided
        self.notification_enabled = self.config.get('notification_settings', {}).get('email_enabled', True)
        self.stock_alert_recipients = self.config.get('notification_settings', {}).get('stock_alert_recipients', [])
        self.email_subject_template = self.config.get('notification_settings', {}).get('email_template_subject', 
                                                                                     'Stock Alert: {product_name} is running low!')
        
    def process_transactions(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a list of daily transactions (sales/purchases)"""
        try:
            # Load current stock data
            if not os.path.exists(self.stock_file):
                logger.error("Stock file not found. Please initialize stock data first.")
                return {'success': False, 'error': 'Stock file not found'}
            
            stock_df = pd.read_csv(self.stock_file)
            
            processed_transactions = []
            stock_alerts = []
            
            for transaction in transactions:
                result = self._process_single_transaction(stock_df, transaction)
                processed_transactions.append(result)
                
                if result.get('stock_alert'):
                    stock_alerts.append(result['stock_alert'])
            
            # Save updated stock data
            stock_df.to_csv(self.stock_file, index=False)
            
            # Send email notifications for stock alerts
            if stock_alerts and self.notification_enabled and self.stock_alert_recipients:
                self._send_stock_alerts(stock_alerts)
            
            # Log transactions
            self._log_transactions(transactions)
            
            # Update sales data for sales transactions
            self._update_sales_data(transactions)
            
            logger.info(f"Processed {len(transactions)} transactions successfully")
            
            return {
                'success': True,
                'transactions_processed': len(transactions),
                'stock_alerts': stock_alerts,
                'processed_details': processed_transactions
            }
            
        except Exception as e:
            logger.error(f"Error processing transactions: {e}")
            return {'success': False, 'error': str(e)}
    
    def _process_single_transaction(self, stock_df: pd.DataFrame, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single transaction and update stock"""
        product_id = transaction['product_id']
        quantity = transaction['quantity']
        trans_type = transaction['type']
        
        # Find product in stock
        product_mask = stock_df['product_id'] == product_id
        
        if not product_mask.any():
            return {
                'success': False,
                'error': f'Product {product_id} not found in inventory',
                'transaction': transaction
            }
        
        # Get current stock level
        current_stock = stock_df.loc[product_mask, 'current_stock'].iloc[0]
        product_name = stock_df.loc[product_mask, 'product_name'].iloc[0]
        reorder_point = stock_df.loc[product_mask, 'reorder_point'].iloc[0]
        
        # Calculate new stock level
        if trans_type == 'sale':
            if current_stock < quantity:
                return {
                    'success': False,
                    'error': f'Insufficient stock for {product_name}. Available: {current_stock}, Required: {quantity}',
                    'transaction': transaction
                }
            new_stock = current_stock - quantity
        elif trans_type == 'purchase':
            new_stock = current_stock + quantity
            # Update unit cost if provided
            if transaction.get('unit_cost'):
                stock_df.loc[product_mask, 'unit_cost'] = transaction['unit_cost']
        else:
            return {
                'success': False,
                'error': f'Unknown transaction type: {trans_type}',
                'transaction': transaction
            }
        
        # Update stock level
        stock_df.loc[product_mask, 'current_stock'] = new_stock
        stock_df.loc[product_mask, 'last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Check if stock alert is needed
        stock_alert = None
        if new_stock <= reorder_point:
            stock_alert = {
                'product_id': product_id,
                'product_name': product_name,
                'current_stock': new_stock,
                'threshold': reorder_point
            }
        
        return {
            'success': True,
            'product_id': product_id,
            'product_name': product_name,
            'transaction_type': trans_type,
            'quantity': quantity,
            'previous_stock': current_stock,
            'new_stock': new_stock,
            'stock_alert': stock_alert
        }
    
    def _log_transactions(self, transactions: List[Dict[str, Any]]) -> None:
        """Log transactions to daily transactions file"""
        try:
            # Load existing transactions
            if os.path.exists(self.transactions_log):
                with open(self.transactions_log, 'r') as f:
                    all_transactions = json.load(f)
            else:
                all_transactions = []
            
            # Add new transactions
            for transaction in transactions:
                transaction_record = {
                    'timestamp': transaction['timestamp'].isoformat() if isinstance(transaction['timestamp'], datetime) else transaction['timestamp'],
                    'type': transaction['type'],
                    'product_id': transaction['product_id'],
                    'quantity': transaction['quantity'],
                    'unit_cost': transaction.get('unit_cost'),
                    'date': datetime.now().strftime('%Y-%m-%d')
                }
                all_transactions.append(transaction_record)
            
            # Keep only last 1000 transactions
            all_transactions = all_transactions[-1000:]
            
            # Save back to file
            os.makedirs(os.path.dirname(self.transactions_log), exist_ok=True)
            with open(self.transactions_log, 'w') as f:
                json.dump(all_transactions, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error logging transactions: {e}")
    
    def _update_sales_data(self, transactions: List[Dict[str, Any]]) -> None:
        """Update sales data file with sales transactions"""
        try:
            # Filter only sales transactions
            sales_transactions = [t for t in transactions if t['type'] == 'sale']
            
            if not sales_transactions:
                return
            
            # Load existing sales data
            if os.path.exists(self.sales_file):
                sales_df = pd.read_csv(self.sales_file)
            else:
                sales_df = pd.DataFrame(columns=['date', 'product_id', 'quantity_sold', 'price'])
            
            # Add new sales records
            new_sales = []
            for transaction in sales_transactions:
                # Get product price from stock file
                stock_df = pd.read_csv(self.stock_file)
                product_mask = stock_df['product_id'] == transaction['product_id']
                
                if product_mask.any():
                    # Estimate price (you might want to add actual selling price to transactions)
                    unit_cost = stock_df.loc[product_mask, 'unit_cost'].iloc[0]
                    estimated_price = unit_cost * 1.3  # Assume 30% markup
                    
                    new_sales.append({
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'product_id': transaction['product_id'],
                        'quantity_sold': transaction['quantity'],
                        'price': estimated_price
                    })
            
            if new_sales:
                new_sales_df = pd.DataFrame(new_sales)
                sales_df = pd.concat([sales_df, new_sales_df], ignore_index=True)
                
                # Save updated sales data
                os.makedirs(os.path.dirname(self.sales_file), exist_ok=True)
                sales_df.to_csv(self.sales_file, index=False)
                
        except Exception as e:
            logger.error(f"Error updating sales data: {e}")
    
    def get_low_stock_items(self, threshold: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get items that are below stock threshold"""
        try:
            if not os.path.exists(self.stock_file):
                return []
            
            stock_df = pd.read_csv(self.stock_file)
            
            if threshold is None:
                # Use individual product reorder points
                low_stock_mask = stock_df['current_stock'] <= stock_df['reorder_point']
            else:
                # Use global threshold
                low_stock_mask = stock_df['current_stock'] <= threshold
            
            low_stock_items = []
            for _, row in stock_df[low_stock_mask].iterrows():
                low_stock_items.append({
                    'product_id': row['product_id'],
                    'product_name': row['product_name'],
                    'current_stock': row['current_stock'],
                    'threshold': row['reorder_point']
                })
            
            return low_stock_items
            
        except Exception as e:
            logger.error(f"Error getting low stock items: {e}")
            return []
    
    def get_stock_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all stock items"""
        try:
            if not os.path.exists(self.stock_file):
                return []
            
            stock_df = pd.read_csv(self.stock_file)
            
            summary = []
            for _, row in stock_df.iterrows():
                summary.append({
                    'product_id': row['product_id'],
                    'product_name': row['product_name'],
                    'current_stock': row['current_stock'],
                    'reorder_point': row['reorder_point'],
                    'unit_cost': row['unit_cost'],
                    'last_updated': row['last_updated']
                })
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting stock summary: {e}")
            return []
    
    def update_product_thresholds(self, product_id: str, reorder_point: int, max_stock: Optional[int] = None) -> bool:
        """Update threshold values for a specific product"""
        try:
            if not os.path.exists(self.stock_file):
                return False
            
            stock_df = pd.read_csv(self.stock_file)
            product_mask = stock_df['product_id'] == product_id
            
            if not product_mask.any():
                logger.error(f"Product {product_id} not found")
                return False
            
            # Update reorder point
            stock_df.loc[product_mask, 'reorder_point'] = reorder_point
            
            # Update max stock if provided
            if max_stock is not None:
                stock_df.loc[product_mask, 'max_stock'] = max_stock
            
            # Update last modified
            stock_df.loc[product_mask, 'last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Save changes
            stock_df.to_csv(self.stock_file, index=False)
            
            logger.info(f"Updated thresholds for product {product_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating product thresholds: {e}")
            return False
    
    def _send_stock_alerts(self, stock_alerts: List[Dict[str, Any]]) -> None:
        """Send email notifications for stock alerts"""
        try:
            for alert in stock_alerts:
                success = self.notification_system.send_stock_alert(
                    alert, 
                    self.stock_alert_recipients, 
                    self.email_subject_template
                )
                
                if success:
                    logger.info(f"Stock alert email sent for {alert['product_name']}")
                else:
                    logger.warning(f"Failed to send stock alert email for {alert['product_name']}")
                    
        except Exception as e:
            logger.error(f"Error sending stock alert emails: {e}")
