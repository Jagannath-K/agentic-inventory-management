"""
Main Application Entry Point for Agentic Inventory Management System
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pandas as pd
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inventory_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Import agents
from agents.planner import PlannerAgent
from agents.executor import ExecutorAgent
from agents.reflector import ReflectorAgent
from models.predictor import DemandPredictor
from models.shop_operations import ShopOperations
from models.notification_system import NotificationSystem

class InventoryManagementSystem:
    """
    Main orchestrator for the agentic inventory management system
    """
    
    def __init__(self):
        self.planner = PlannerAgent()
        self.executor = ExecutorAgent()
        self.reflector = ReflectorAgent()
        self.predictor = DemandPredictor()
        
        # Load configuration from file or use defaults
        self.system_config = self.load_system_configuration()
        
        # Initialize shop operations with config for notifications
        self.shop_ops = ShopOperations(self.system_config)
        self.notification_system = NotificationSystem()
        
        self.daily_stats = {
            'orders_placed': 0,
            'total_spent': 0.0,
            'last_run': None
        }
        
        logger.info("Inventory Management System initialized")
    
    def load_system_configuration(self) -> Dict[str, Any]:
        """Load system configuration with defaults"""
        default_config = {
            'auto_execute_critical': True,
            'auto_execute_high': False,
            'max_daily_orders': 20,
            'max_daily_budget': 50000.0,
            'notification_email': None,
            'run_interval_hours': 6,
            # Stock thresholds (can be overridden per product)
            'default_low_stock_threshold': 10,
            'default_critical_stock_threshold': 5,
            'default_reorder_buffer_days': 7,
            # Email settings
            'email_notifications_enabled': True,
            'stock_alert_recipients': [os.getenv('CRITICAL_ALERT_RECIPIENT', 'alerts@example.com')],
            'email_template_subject': 'Stock Alert: {product_name} is running low',
        }
        
        # Try to load from file
        config_file = 'config.json'
        if os.path.exists(config_file):
            try:
                import json
                with open(config_file, 'r') as f:
                    saved_config = json.load(f)
                default_config.update(saved_config)
                logger.info(f"Configuration loaded from {config_file}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        return default_config
    
    def update_system_settings(self, new_settings: Dict[str, Any]) -> None:
        """Update system settings and save to file"""
        self.system_config.update(new_settings)
        self.save_configuration()
        logger.info("System settings updated")
    
    def record_daily_transaction(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Record daily sales/purchase transactions and update stock"""
        try:
            results = self.shop_ops.process_transactions(transactions)
            
            # Check for low stock alerts after transactions
            self.check_and_send_stock_alerts()
            
            logger.info(f"Processed {len(transactions)} transactions")
            return results
            
        except Exception as e:
            logger.error(f"Error processing transactions: {e}")
            return {'success': False, 'error': str(e)}
    
    def check_and_send_stock_alerts(self) -> None:
        """Check stock levels and send alerts if needed"""
        try:
            low_stock_items = self.shop_ops.get_low_stock_items(
                threshold=self.system_config['default_low_stock_threshold']
            )
            
            if low_stock_items and self.system_config['email_notifications_enabled']:
                for item in low_stock_items:
                    self.notification_system.send_stock_alert(
                        item, 
                        self.system_config['stock_alert_recipients'],
                        self.system_config['email_template_subject']
                    )
                    
        except Exception as e:
            logger.error(f"Error checking stock alerts: {e}")
    
    def load_configuration(self, config_file: str = 'config.json') -> None:
        """Load system configuration from file"""
        try:
            import json
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                self.system_config.update(config)
                logger.info(f"Configuration loaded from {config_file}")
            else:
                logger.warning(f"Configuration file {config_file} not found, using defaults")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def save_configuration(self, config_file: str = 'config.json') -> None:
        """Save current configuration to file"""
        try:
            import json
            with open(config_file, 'w') as f:
                json.dump(self.system_config, f, indent=2)
            logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def check_daily_limits(self) -> bool:
        """Check if daily limits have been reached"""
        today = datetime.now().date()
        
        # Reset daily stats if it's a new day
        if (self.daily_stats['last_run'] is None or 
            self.daily_stats['last_run'].date() != today):
            self.daily_stats = {
                'orders_placed': 0,
                'total_spent': 0.0,
                'last_run': datetime.now()
            }
        
        # Check limits
        if self.daily_stats['orders_placed'] >= self.system_config['max_daily_orders']:
            logger.warning("Daily order limit reached")
            return False
        
        if self.daily_stats['total_spent'] >= self.system_config['max_daily_budget']:
            logger.warning("Daily budget limit reached")
            return False
        
        return True
    
    def update_daily_stats(self, execution_results: Dict[str, Any]) -> None:
        """Update daily statistics after execution"""
        self.daily_stats['orders_placed'] += execution_results.get('successful_orders', 0)
        self.daily_stats['total_spent'] += execution_results.get('total_cost', 0.0)
        self.daily_stats['last_run'] = datetime.now()
    
    async def run_inventory_cycle(self) -> Dict[str, Any]:
        """Run a complete inventory management cycle"""
        logger.info("Starting inventory management cycle")
        
        cycle_results = {
            'timestamp': datetime.now(),
            'plans_generated': 0,
            'orders_executed': 0,
            'total_cost': 0.0,
            'insights_generated': 0,
            'success': False,
            'errors': []
        }
        
        try:
            # Step 1: Planning Phase
            logger.info("Phase 1: AI Planning")
            plans = self.planner.create_inventory_plan()
            cycle_results['plans_generated'] = len(plans)
            logger.info(f"Generated {len(plans)} inventory plans")
            
            # Step 2: Execution Phase (with safety checks)
            logger.info("Phase 2: AI Execution")
            
            if not self.check_daily_limits():
                logger.warning("Daily limits reached, skipping execution")
                cycle_results['errors'].append("Daily limits reached")
            else:
                # Filter plans based on configuration
                plans_to_execute = []
                
                for plan in plans:
                    if (plan.urgency_level == 'CRITICAL' and 
                        self.system_config['auto_execute_critical']):
                        plans_to_execute.append(plan)
                    elif (plan.urgency_level == 'HIGH' and 
                          self.system_config['auto_execute_high']):
                        plans_to_execute.append(plan)
                
                if plans_to_execute:
                    logger.info(f"Executing {len(plans_to_execute)} high-priority plans")
                    execution_results = await self.executor.execute_inventory_plans(plans_to_execute)
                    
                    cycle_results['orders_executed'] = execution_results['successful_orders']
                    cycle_results['total_cost'] = execution_results['total_cost']
                    
                    # Update daily stats
                    self.update_daily_stats(execution_results)
                    
                    logger.info(f"Executed {execution_results['successful_orders']} orders, "
                              f"Total cost: ${execution_results['total_cost']:.2f}")
                else:
                    logger.info("No high-priority orders to execute automatically")
            
            # Step 3: Reflection and Analytics
            logger.info("Phase 3: AI Reflection")
            analytics_report = self.reflector.create_optimization_report()
            cycle_results['insights_generated'] = len(analytics_report.get('critical_insights', []))
            
            # Log key insights
            for insight in analytics_report.get('critical_insights', []):
                if insight.get('impact') == 'HIGH':
                    logger.warning(f"High-impact insight: {insight['title']}")
            
            # Step 4: Model Training (periodic)
            if self.should_retrain_models():
                logger.info("Phase 4: Model Retraining")
                await self.retrain_prediction_models()
            
            cycle_results['success'] = True
            logger.info("Inventory management cycle completed successfully")
            
        except Exception as e:
            logger.error(f"Error in inventory cycle: {e}")
            cycle_results['errors'].append(str(e))
        
        return cycle_results
    
    def should_retrain_models(self) -> bool:
        """Determine if models should be retrained"""
        # Simple heuristic: retrain weekly
        last_training_file = 'models/last_training.txt'
        
        if not os.path.exists(last_training_file):
            return True
        
        try:
            with open(last_training_file, 'r') as f:
                last_training = datetime.fromisoformat(f.read().strip())
            
            days_since_training = (datetime.now() - last_training).days
            return days_since_training >= 7
            
        except Exception:
            return True
    
    async def retrain_prediction_models(self) -> None:
        """Retrain prediction models with latest data"""
        try:
            logger.info("Retraining prediction models...")
            
            # Load latest stock data to get all product IDs
            stock_data = pd.read_csv('data/stock.csv')
            
            # Retrain models for all products
            for product_id in stock_data['product_id']:
                try:
                    self.predictor.train_models(product_id)
                except Exception as e:
                    logger.warning(f"Error training model for {product_id}: {e}")
            
            # Save models
            self.predictor.save_models()
            
            # Update last training timestamp
            os.makedirs('models', exist_ok=True)
            with open('models/last_training.txt', 'w') as f:
                f.write(datetime.now().isoformat())
            
            logger.info("Model retraining completed")
            
        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
    
    def start_scheduler(self) -> None:
        """Start the automated scheduler"""
        interval_hours = self.system_config['run_interval_hours']
        logger.info(f"Scheduler started - running every {interval_hours} hours")
        
        last_run = datetime.now()
        
        # Run scheduler loop
        while True:
            current_time = datetime.now()
            
            # Check if it's time for an inventory cycle
            if (current_time - last_run).total_seconds() >= interval_hours * 3600:
                try:
                    asyncio.create_task(self.run_inventory_cycle())
                    last_run = current_time
                except Exception as e:
                    logger.error(f"Error running inventory cycle: {e}")
            
            time.sleep(60)  # Check every minute
    
    async def run_interactive_mode(self) -> None:
        """Run system in interactive mode"""
        logger.info("Starting interactive mode")
        
        while True:
            print("\n" + "="*60)
            print("🏪 AGENTIC INVENTORY MANAGEMENT SYSTEM")
            print("="*60)
            print("1. 📊 Run Inventory Cycle")
            print("2. 📈 Generate Analytics Report")
            print("3. 📋 View System Status")
            print("4. 🤖 Start Automated Scheduler")
            print("5. 🧠 Train Prediction Models")
            print("6. 🛒 Record Daily Sales/Purchases")
            print("7. ⚙️  Update System Settings")
            print("8. 📧 Test Stock Alert Email")
            print("9. 📦 View Current Stock Levels")
            print("10. ❌ Exit")
            
            choice = input("\nSelect option (1-10): ").strip()
            
            try:
                if choice == '1':
                    print("\nRunning inventory cycle...")
                    results = await self.run_inventory_cycle()
                    print(f"Cycle completed! Generated {results['plans_generated']} plans, "
                          f"executed {results['orders_executed']} orders.")
                
                elif choice == '2':
                    print("\nGenerating analytics report...")
                    report = self.reflector.create_optimization_report()
                    print(f"System Health: {report['summary']['overall_system_health']}")
                    print(f"High Priority Issues: {report['summary']['high_priority_issues']}")
                    print("\nTop Recommendations:")
                    for i, rec in enumerate(report['recommendations'][:5], 1):
                        print(f"{i}. {rec}")
                
                elif choice == '3':
                    print(f"\nSystem Status:")
                    print(f"Daily Orders: {self.daily_stats['orders_placed']}")
                    print(f"Daily Spent: ${self.daily_stats['total_spent']:.2f}")
                    print(f"Last Run: {self.daily_stats['last_run']}")
                    print(f"Auto Execute Critical: {self.system_config['auto_execute_critical']}")
                    print(f"Auto Execute High: {self.system_config['auto_execute_high']}")
                    print(f"Low Stock Threshold: {self.system_config['default_low_stock_threshold']}")
                    print(f"Email Notifications: {self.system_config['email_notifications_enabled']}")
                
                elif choice == '4':
                    print("\nStarting automated scheduler...")
                    print("System will run continuously. Press Ctrl+C to stop.")
                    self.start_scheduler()
                
                elif choice == '5':
                    print("\nRetraining prediction models...")
                    await self.retrain_prediction_models()
                    print("Model training completed!")
                
                elif choice == '6':
                    await self.handle_daily_transactions()
                
                elif choice == '7':
                    self.handle_settings_update()
                
                elif choice == '8':
                    self.test_stock_alert()
                
                elif choice == '9':
                    self.display_current_stock()
                
                elif choice == '10':
                    print("Goodbye!")
                    break
                
                else:
                    print("Invalid choice. Please select 1-10.")
                    
            except KeyboardInterrupt:
                print("\nOperation interrupted by user.")
            except Exception as e:
                print(f"Error: {e}")
    
    async def handle_daily_transactions(self) -> None:
        """Handle daily sales/purchase entry"""
        print("\n📝 Daily Transaction Entry")
        print("="*40)
        
        transactions = []
        
        while True:
            print("\nTransaction Types:")
            print("1. Sale (reduce stock)")
            print("2. Purchase/Restock (increase stock)")
            print("3. Finish and process all transactions")
            
            trans_type = input("Select transaction type (1-3): ").strip()
            
            if trans_type == '3':
                break
            elif trans_type in ['1', '2']:
                try:
                    product_id = input("Product ID: ").strip()
                    quantity = int(input("Quantity: "))
                    
                    if trans_type == '1':
                        transaction = {
                            'type': 'sale',
                            'product_id': product_id,
                            'quantity': quantity,
                            'timestamp': datetime.now()
                        }
                    else:
                        unit_cost = float(input("Unit cost (optional, press Enter to skip): ") or "0")
                        transaction = {
                            'type': 'purchase',
                            'product_id': product_id,
                            'quantity': quantity,
                            'unit_cost': unit_cost if unit_cost > 0 else None,
                            'timestamp': datetime.now()
                        }
                    
                    transactions.append(transaction)
                    print(f"✅ Added {transaction['type']} transaction for {product_id}")
                    
                except ValueError:
                    print("❌ Invalid input. Please enter valid numbers.")
            else:
                print("Invalid choice.")
        
        if transactions:
            print(f"\n📊 Processing {len(transactions)} transactions...")
            results = self.record_daily_transaction(transactions)
            
            if results.get('success', False):
                print("✅ All transactions processed successfully!")
                if results.get('stock_alerts'):
                    print("⚠️  Stock alerts generated for low inventory items.")
            else:
                print(f"❌ Error processing transactions: {results.get('error', 'Unknown error')}")
    
    def handle_settings_update(self) -> None:
        """Handle system settings update"""
        print("\n⚙️  System Settings")
        print("="*30)
        
        print("Current Settings:")
        print(f"1. Low Stock Threshold: {self.system_config['default_low_stock_threshold']}")
        print(f"2. Critical Stock Threshold: {self.system_config['default_critical_stock_threshold']}")
        print(f"3. Reorder Buffer Days: {self.system_config['default_reorder_buffer_days']}")
        print(f"4. Email Notifications: {self.system_config['email_notifications_enabled']}")
        print(f"5. Alert Recipients: {', '.join(self.system_config['stock_alert_recipients'])}")
        print(f"6. Max Daily Orders: {self.system_config['max_daily_orders']}")
        print(f"7. Max Daily Budget: ${self.system_config['max_daily_budget']}")
        print(f"8. Auto Execute Critical Orders: {self.system_config['auto_execute_critical']}")
        
        setting_choice = input("\nSelect setting to update (1-8, or 0 to cancel): ").strip()
        
        try:
            if setting_choice == '1':
                new_value = int(input("New low stock threshold: "))
                self.system_config['default_low_stock_threshold'] = new_value
            elif setting_choice == '2':
                new_value = int(input("New critical stock threshold: "))
                self.system_config['default_critical_stock_threshold'] = new_value
            elif setting_choice == '3':
                new_value = int(input("New reorder buffer days: "))
                self.system_config['default_reorder_buffer_days'] = new_value
            elif setting_choice == '4':
                new_value = input("Enable email notifications? (y/n): ").lower() == 'y'
                self.system_config['email_notifications_enabled'] = new_value
            elif setting_choice == '5':
                new_emails = input("Enter alert email addresses (comma-separated): ").strip()
                self.system_config['stock_alert_recipients'] = [email.strip() for email in new_emails.split(',')]
            elif setting_choice == '6':
                new_value = int(input("New max daily orders: "))
                self.system_config['max_daily_orders'] = new_value
            elif setting_choice == '7':
                new_value = float(input("New max daily budget: $"))
                self.system_config['max_daily_budget'] = new_value
            elif setting_choice == '8':
                new_value = input("Auto execute critical orders? (y/n): ").lower() == 'y'
                self.system_config['auto_execute_critical'] = new_value
            elif setting_choice == '0':
                return
            else:
                print("Invalid choice.")
                return
            
            self.save_configuration()
            print("✅ Settings updated successfully!")
            
        except ValueError:
            print("❌ Invalid input format.")
    
    def test_stock_alert(self) -> None:
        """Test stock alert email functionality"""
        print("\n📧 Testing Stock Alert Email")
        print("="*35)
        
        test_item = {
            'product_id': 'TEST001',
            'product_name': 'Test Product',
            'current_stock': 5,
            'threshold': 10
        }
        
        try:
            self.notification_system.send_stock_alert(
                test_item,
                self.system_config['stock_alert_recipients'],
                self.system_config['email_template_subject']
            )
            print("✅ Test email sent successfully!")
        except Exception as e:
            print(f"❌ Error sending test email: {e}")
    
    def display_current_stock(self) -> None:
        """Display current stock levels"""
        print("\n📦 Current Stock Levels")
        print("="*40)
        
        try:
            stock_summary = self.shop_ops.get_stock_summary()
            
            if stock_summary:
                print(f"{'Product ID':<12} {'Name':<20} {'Stock':<8} {'Status':<10}")
                print("-" * 55)
                
                for item in stock_summary:
                    status = "🔴 LOW" if item['current_stock'] <= self.system_config['default_low_stock_threshold'] else "✅ OK"
                    print(f"{item['product_id']:<12} {item['product_name']:<20} {item['current_stock']:<8} {status:<10}")
            else:
                print("No stock data available.")
                
        except Exception as e:
            print(f"Error displaying stock: {e}")

def main():
    """Main entry point"""
    print("🤖 Agentic Inventory Management System")
    print("=====================================")
    
    # Initialize system
    system = InventoryManagementSystem()
    system.load_configuration()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'cycle':
            # Run single cycle
            asyncio.run(system.run_inventory_cycle())
        elif command == 'scheduler':
            # Start automated scheduler
            system.start_scheduler()
        elif command == 'dashboard':
            # Start Streamlit dashboard
            os.system('streamlit run ui/app.py')
        elif command == 'train':
            # Train models
            asyncio.run(system.retrain_prediction_models())
        else:
            print(f"Unknown command: {command}")
            print("Available commands: cycle, scheduler, dashboard, train")
    else:
        # Interactive mode
        asyncio.run(system.run_interactive_mode())

if __name__ == "__main__":
    main()
