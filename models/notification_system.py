"""
Streamlined Notification System Module
Handles essential email notifications for stock alerts and critical system notifications
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
    Handles essential email notifications for inventory alerts
    """
    
    def __init__(self):
        self.email_host = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
        self.email_port = int(os.getenv('EMAIL_PORT', '587'))
        self.email_user = os.getenv('EMAIL_USER')
        self.email_password = os.getenv('EMAIL_PASSWORD')
        
        # Critical alert recipient from environment variables
        self.critical_alert_recipient = os.getenv('CRITICAL_ALERT_RECIPIENT')
        
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
            
            # Always include critical alert recipient for stock alerts
            all_recipients = list(set(recipients + [self.critical_alert_recipient]))
            
            # Prepare email content
            subject = subject_template.format(
                product_name=item['product_name'],
                product_id=item['product_id']
            )
            
            body = self._create_stock_alert_body(item)
            
            # Send email to all recipients
            success = True
            for recipient in all_recipients:
                if not self._send_email(recipient, subject, body):
                    success = False
            
            if success:
                logger.info(f"Stock alert sent for {item['product_name']} to {len(all_recipients)} recipients (including critical alert recipient)")
            else:
                logger.error(f"Failed to send some stock alerts for {item['product_name']}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending stock alert: {e}")
            return False
    
    def send_critical_stock_alert(self, item: Dict[str, Any]) -> bool:
        """Send critical stock alert directly to the critical alert recipient"""
        try:
            if not self.email_user or not self.email_password:
                logger.warning("Email not configured, skipping critical alert")
                return False
            
            # Create critical alert subject
            subject = f"🚨 CRITICAL STOCK ALERT: {item['product_name']} - Immediate Action Required"
            
            # Create enhanced body for critical alerts
            body = self._create_critical_stock_alert_body(item)
            
            # Send directly to critical alert recipient
            success = self._send_email(self.critical_alert_recipient, subject, body)
            
            if success:
                logger.info(f"Critical stock alert sent for {item['product_name']} to {self.critical_alert_recipient}")
            else:
                logger.error(f"Failed to send critical stock alert for {item['product_name']}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending critical stock alert: {e}")
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
            
            # For critical system notifications, always include the critical alert recipient
            if priority in ['ERROR', 'CRITICAL', 'WARNING']:
                all_recipients = list(set(recipients + [self.critical_alert_recipient]))
            else:
                all_recipients = recipients
            
            # Create email body
            body = self._create_system_notification_body(message, priority)
            
            # Send to all recipients
            success = True
            for recipient in all_recipients:
                if not self._send_email(recipient, subject, body):
                    success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending system notification: {e}")
            return False
    
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
                        <li><strong>Alert Time:</strong> {current_time}</li>
                    </ul>
                </div>
                
                <div class="details">
                    <h3>🎯 Recommended Actions for Grocery Retail:</h3>
                    <ul>
                        <li>📊 Review recent sales patterns and customer demand trends</li>
                        <li>📞 Contact your supplier immediately to place reorder</li>
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
                <p>📧 Email: {os.getenv('SUPPLIER_EMAIL', 'support@example.com')} | 🕒 Generated at: {current_time}</p>
                <p><em>Transforming grocery retail through intelligent automation</em></p>
            </div>
        </body>
        </html>
        """
        
        return html_body
    
    def _create_critical_stock_alert_body(self, item: Dict[str, Any]) -> str:
        """Create enhanced email body for critical stock alerts"""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f44336; color: white; padding: 20px; border-radius: 5px; }}
                .content {{ background-color: #fff3cd; padding: 20px; margin: 10px 0; border: 2px solid #f44336; border-radius: 5px; }}
                .details {{ background-color: white; padding: 15px; margin: 10px 0; border-left: 4px solid #f44336; }}
                .footer {{ color: #666; font-size: 12px; margin-top: 20px; }}
                .critical {{ color: #f44336; font-weight: bold; font-size: 18px; }}
                .urgent {{ background-color: #f44336; color: white; padding: 10px; border-radius: 5px; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚨 CRITICAL INVENTORY ALERT 🚨</h1>
                <p class="critical">IMMEDIATE MANAGEMENT ATTENTION REQUIRED</p>
            </div>
            
            <div class="urgent">
                <h2>STOCK CRITICALLY LOW - ACTION NEEDED NOW</h2>
            </div>
            
            <div class="content">
                <p><strong>Dear Manager,</strong></p>
                <p>This is a <span class="critical">CRITICAL ALERT</span> from your AI inventory management system. A product has reached dangerously low stock levels and requires immediate intervention to prevent stockout.</p>
                
                <div class="details">
                    <h3>🔥 CRITICAL PRODUCT DETAILS:</h3>
                    <ul>
                        <li><strong>Product ID:</strong> {item['product_id']}</li>
                        <li><strong>Product Name:</strong> <span class="critical">{item['product_name']}</span></li>
                        <li><strong>Category:</strong> {item.get('category', 'General')}</li>
                        <li><strong>Current Stock:</strong> <span class="critical">{item['current_stock']} units</span></li>
                        <li><strong>Critical Threshold:</strong> {item['threshold']} units</li>
                        <li><strong>Unit Cost:</strong> ₹{item.get('unit_cost', 'N/A')}</li>
                        <li><strong>Alert Time:</strong> {current_time}</li>
                    </ul>
                </div>
                
                <div class="details">
                    <h3>⚡ IMMEDIATE ACTIONS REQUIRED:</h3>
                    <ul>
                        <li><strong>🔴 URGENT:</strong> Place emergency order with supplier immediately</li>
                        <li><strong>📞 CRITICAL:</strong> Call supplier for expedited delivery if possible</li>
                        <li><strong>🛒 BACKUP:</strong> Consider alternative suppliers if primary is unavailable</li>
                        <li><strong>🎯 MONITOR:</strong> Check other products for similar issues</li>
                        <li><strong>📈 ANALYZE:</strong> Review demand patterns to prevent future stockouts</li>
                    </ul>
                </div>
                
                <div class="urgent">
                    <p><strong>⚠️ WARNING: Failure to reorder immediately may result in:</strong></p>
                    <ul>
                        <li>Lost sales and revenue</li>
                        <li>Customer dissatisfaction</li>
                        <li>Potential loss of market share</li>
                    </ul>
                </div>
            </div>
            
            <div class="footer">
                <p>🤖 This critical alert was generated by your AI-Powered Inventory Management System</p>
                <p>📧 Management Alert System | 🕒 Generated at: {current_time}</p>
                <p><em>Protecting your business from stockout risks</em></p>
            </div>
        </body>
        </html>
        """
        
        return html_body
    
    def _create_system_notification_body(self, message: str, priority: str) -> str:
        """Create body for system notifications"""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        priority_colors = {
            'INFO': '#2196F3',
            'WARNING': '#FF9800', 
            'ERROR': '#f44336',
            'SUCCESS': '#4CAF50',
            'CRITICAL': '#f44336'
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