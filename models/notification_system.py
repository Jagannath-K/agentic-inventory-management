"""
Notification System Module
Handles email notifications for stock alerts and other system notifications
"""

import smtplib
import os
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class NotificationSystem:
    """
    Handles email notifications for inventory alerts
    """
    
    def __init__(self):
        self.email_host = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
        self.email_port = int(os.getenv('EMAIL_PORT', '587'))
        self.email_user = os.getenv('EMAIL_USER')
        self.email_password = os.getenv('EMAIL_PASSWORD')
        
        # Verify email configuration
        if not self.email_user or not self.email_password:
            logger.warning("Email credentials not configured. Please set EMAIL_USER and EMAIL_PASSWORD in .env file")
    
    def send_stock_alert(self, 
                        item: Dict[str, Any], 
                        recipients: List[str], 
                        subject_template: str) -> bool:
        """Send stock alert email for a low-stock item"""
        try:
            if not self.email_user or not self.email_password:
                logger.warning("Email not configured, skipping notification")
                return False
            
            # Prepare email content
            subject = subject_template.format(
                product_name=item['product_name'],
                product_id=item['product_id']
            )
            
            body = self._create_stock_alert_body(item)
            
            # Send email to all recipients
            success = True
            for recipient in recipients:
                if not self._send_email(recipient, subject, body):
                    success = False
            
            if success:
                logger.info(f"Stock alert sent for {item['product_name']} to {len(recipients)} recipients")
            else:
                logger.error(f"Failed to send some stock alerts for {item['product_name']}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending stock alert: {e}")
            return False
    
    def _create_stock_alert_body(self, item: Dict[str, Any]) -> str:
        """Create the email body for grocery retail stock alert"""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #2196F3; color: white; padding: 15px; border-radius: 5px; }}
                .content {{ background-color: #f9f9f9; padding: 20px; margin: 10px 0; border-radius: 5px; }}
                .details {{ background-color: white; padding: 15px; margin: 10px 0; border-left: 4px solid #2196F3; }}
                .footer {{ color: #666; font-size: 12px; margin-top: 20px; }}
                .urgent {{ color: #f44336; font-weight: bold; }}
                .grocery {{ color: #4CAF50; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>🏪 GROCERY RETAIL INVENTORY ALERT</h2>
            </div>
            
            <div class="content">
                <p>Dear Store Manager,</p>
                <p>This is an automated alert from your AI-powered grocery inventory management system. One of your products has reached the low stock threshold and requires immediate attention.</p>
                
                <div class="details">
                    <h3>📦 Product Details:</h3>
                    <ul>
                        <li><strong>Product ID:</strong> {item['product_id']}</li>
                        <li><strong>Product Name:</strong> <span class="grocery">{item['product_name']}</span></li>
                        <li><strong>Category:</strong> {item.get('category', 'General')}</li>
                        <li><strong>Current Stock:</strong> <span class="urgent">{item['current_stock']} units</span></li>
                        <li><strong>Reorder Threshold:</strong> {item['threshold']} units</li>
                        <li><strong>Unit Cost:</strong> ₹{item.get('unit_cost', 'N/A')}</li>
                        <li><strong>Supplier ID:</strong> {item['supplier_id']}</li>
                        <li><strong>Alert Time:</strong> {current_time}</li>
                    </ul>
                </div>
                
                <div class="details">
                    <h3>🎯 Recommended Actions for Grocery Retail:</h3>
                    <ul>
                        <li>📊 Review recent sales patterns and customer demand trends</li>
                        <li>📞 Contact supplier {item['supplier_id']} immediately to place reorder</li>
                        <li>🧮 Consider bulk purchasing for better margins (if applicable)</li>
                        <li>📅 Check for seasonal demand patterns (festivals, month-end shopping)</li>
                        <li>🔍 Monitor competitor pricing for this category</li>
                        <li>⚡ Ensure no customer disappointment due to stockout</li>
                    </ul>
                </div>
                
                <div class="details">
                    <h3>🤖 AI Insights:</h3>
                    <p><em>This alert was generated by analyzing historical sales data, seasonal patterns, and current inventory levels. The AI system has detected that immediate action is required to maintain optimal stock levels.</em></p>
                </div>
                
                <p><strong>⚡ Immediate Action Required:</strong> Please reorder this product as soon as possible to avoid customer disappointment and maintain store reputation.</p>
            </div>
            
            <div class="footer">
                <p>🏪 This alert was generated automatically by your AI-Powered Grocery Inventory Management System.</p>
                <p>📧 Email: jagannath.backup.2005@gmail.com | 🕒 Generated at: {current_time}</p>
                <p><em>Transforming grocery retail through intelligent automation</em></p>
            </div>
        </body>
        </html>
        """
        
        return html_body
    
    def _send_email(self, recipient: str, subject: str, body: str) -> bool:
        """Send email to a single recipient"""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.email_user
            msg['To'] = recipient
            msg['Subject'] = subject
            
            # Add HTML body
            html_part = MIMEText(body, 'html')
            msg.attach(html_part)
            
            # Connect to server and send email
            server = smtplib.SMTP(self.email_host, self.email_port)
            server.starttls()
            server.login(self.email_user, self.email_password)
            
            text = msg.as_string()
            server.sendmail(self.email_user, recipient, text)
            server.quit()
            
            logger.debug(f"Email sent successfully to {recipient}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {recipient}: {e}")
            return False
    
    def send_system_notification(self, 
                                message: str, 
                                subject: str, 
                                recipients: List[str], 
                                priority: str = 'INFO') -> bool:
        """Send general system notification"""
        try:
            if not self.email_user or not self.email_password:
                logger.warning("Email not configured, skipping notification")
                return False
            
            # Create email body
            body = self._create_system_notification_body(message, priority)
            
            # Send to all recipients
            success = True
            for recipient in recipients:
                if not self._send_email(recipient, subject, body):
                    success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending system notification: {e}")
            return False
    
    def _create_system_notification_body(self, message: str, priority: str) -> str:
        """Create body for system notifications"""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        priority_colors = {
            'INFO': '#2196F3',
            'WARNING': '#FF9800', 
            'ERROR': '#f44336',
            'SUCCESS': '#4CAF50'
        }
        
        color = priority_colors.get(priority, '#2196F3')
        
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: {color}; color: white; padding: 15px; border-radius: 5px; }}
                .content {{ background-color: #f9f9f9; padding: 20px; margin: 10px 0; border-radius: 5px; }}
                .footer {{ color: #666; font-size: 12px; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>📊 INVENTORY SYSTEM NOTIFICATION</h2>
                <p>Priority: {priority}</p>
            </div>
            
            <div class="content">
                <p><strong>Message:</strong></p>
                <p>{message}</p>
                
                <p><strong>Timestamp:</strong> {current_time}</p>
            </div>
            
            <div class="footer">
                <p>This notification was generated by your Agentic Inventory Management System.</p>
            </div>
        </body>
        </html>
        """
        
        return html_body
    
    def test_email_configuration(self) -> Dict[str, Any]:
        """Test email configuration"""
        try:
            if not self.email_user or not self.email_password:
                return {
                    'success': False,
                    'error': 'Email credentials not configured'
                }
            
            # Try to connect to SMTP server
            server = smtplib.SMTP(self.email_host, self.email_port)
            server.starttls()
            server.login(self.email_user, self.email_password)
            server.quit()
            
            return {
                'success': True,
                'message': 'Email configuration is working correctly'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Email configuration test failed: {e}'
            }
    
    def send_daily_summary(self, 
                          summary_data: Dict[str, Any], 
                          recipients: List[str]) -> bool:
        """Send daily inventory summary"""
        try:
            subject = f"Daily Inventory Summary - {datetime.now().strftime('%Y-%m-%d')}"
            
            body = self._create_daily_summary_body(summary_data)
            
            success = True
            for recipient in recipients:
                if not self._send_email(recipient, subject, body):
                    success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending daily summary: {e}")
            return False
    
    def _create_daily_summary_body(self, summary_data: Dict[str, Any]) -> str:
        """Create daily summary email body"""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #4CAF50; color: white; padding: 15px; border-radius: 5px; }}
                .content {{ background-color: #f9f9f9; padding: 20px; margin: 10px 0; border-radius: 5px; }}
                .stats {{ background-color: white; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .footer {{ color: #666; font-size: 12px; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>📊 DAILY INVENTORY SUMMARY</h2>
                <p>{current_time}</p>
            </div>
            
            <div class="content">
                <div class="stats">
                    <h3>Today's Activity:</h3>
                    <ul>
                        <li>Transactions Processed: {summary_data.get('transactions_processed', 0)}</li>
                        <li>Orders Executed: {summary_data.get('orders_executed', 0)}</li>
                        <li>Total Spent: ${summary_data.get('total_spent', 0):.2f}</li>
                        <li>Low Stock Alerts: {summary_data.get('low_stock_alerts', 0)}</li>
                    </ul>
                </div>
                
                <div class="stats">
                    <h3>System Health:</h3>
                    <p>Overall Status: {summary_data.get('system_health', 'Good')}</p>
                    <p>Issues Requiring Attention: {summary_data.get('issues_count', 0)}</p>
                </div>
            </div>
            
            <div class="footer">
                <p>Generated by Agentic Inventory Management System</p>
            </div>
        </body>
        </html>
        """
        
        return html_body
