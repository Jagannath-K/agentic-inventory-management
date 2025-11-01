"""
Planner Agent - Strategic inventory planning and demand forecasting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InventoryPlan:
    """Data class for inventory planning recommendations for grocery retail"""
    product_id: str
    product_name: str
    current_stock: int
    predicted_demand: float
    reorder_quantity: int
    reorder_date: datetime
    urgency_level: str
    reasoning: str
    confidence_score: float
    category: str = ""

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__class__.__name__}.{name}")
    
    @abstractmethod
    def process(self, *args, **kwargs):
        """Main processing method - to be implemented by subclasses"""
        pass

class PlannerAgent(BaseAgent):
    """
    AI Agent responsible for strategic grocery inventory planning and demand forecasting
    Specialized for micro enterprise grocery retail management
    """
    
    def __init__(self):
        super().__init__("PlannerAgent")
        self.sales_data = None
        self.stock_data = None
        self.supplier_data = None
        self.seasonal_factors = {}
        
    def load_data(self) -> None:
        """Load all necessary data for planning"""
        try:
            self.sales_data = pd.read_csv('data/sales.csv')
            self.stock_data = pd.read_csv('data/stock.csv')
            
            # Convert date columns with error handling
            self.sales_data['date'] = pd.to_datetime(self.sales_data['date'], errors='coerce')
            self.stock_data['last_updated'] = pd.to_datetime(self.stock_data['last_updated'], errors='coerce')
            
            self.logger.info("Data loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def analyze_demand_patterns(self, product_id: str) -> Dict[str, Any]:
        """Analyze historical demand patterns for a specific product"""
        product_sales = self.sales_data[self.sales_data['product_id'] == product_id]
        
        if product_sales.empty:
            return {
                'average_daily_demand': 0,
                'trend': 'stable',
                'seasonality': 1.0,
                'volatility': 0
            }
        
        # Calculate daily demand
        daily_sales = product_sales.groupby('date')['quantity_sold'].sum()
        
        # Fill missing dates with 0
        date_range = pd.date_range(
            start=daily_sales.index.min(),
            end=daily_sales.index.max(),
            freq='D'
        )
        daily_sales = daily_sales.reindex(date_range, fill_value=0)
        
        # Calculate metrics
        avg_demand = daily_sales.mean()
        volatility = daily_sales.std()
        
        # Simple trend analysis (last 7 days vs previous 7 days)
        recent_demand = daily_sales.tail(7).mean()
        previous_demand = daily_sales.tail(14).head(7).mean()
        
        if recent_demand > previous_demand * 1.1:
            trend = 'increasing'
        elif recent_demand < previous_demand * 0.9:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        # Basic seasonality (day of week effect)
        dow_demand = daily_sales.groupby(daily_sales.index.dayofweek).mean()
        seasonality_factor = dow_demand.max() / dow_demand.mean() if dow_demand.mean() > 0 else 1.0
        
        return {
            'average_daily_demand': avg_demand,
            'trend': trend,
            'seasonality': seasonality_factor,
            'volatility': volatility,
            'recent_trend_factor': recent_demand / previous_demand if previous_demand > 0 else 1.0
        }
    
    def forecast_demand(self, product_id: str, days_ahead: int = 30) -> float:
        """Forecast demand for a product over the specified period"""
        demand_analysis = self.analyze_demand_patterns(product_id)
        
        base_demand = demand_analysis['average_daily_demand'] * days_ahead
        
        # Apply trend factor
        trend_multiplier = 1.0
        if demand_analysis['trend'] == 'increasing':
            trend_multiplier = 1.2
        elif demand_analysis['trend'] == 'decreasing':
            trend_multiplier = 0.8
        
        # Apply recent trend
        recent_trend_factor = demand_analysis['recent_trend_factor']
        
        # Apply seasonality
        seasonality_factor = demand_analysis['seasonality']
        
        # Calculate forecasted demand
        forecasted_demand = base_demand * trend_multiplier * recent_trend_factor * seasonality_factor
        
        # Add safety buffer based on volatility
        safety_buffer = demand_analysis['volatility'] * np.sqrt(days_ahead) * 1.65  # 95% service level
        
        return max(0, forecasted_demand + safety_buffer)
    
    def calculate_optimal_reorder_point(self, product_id: str) -> Tuple[int, str]:
        """Calculate optimal reorder point and timing"""
        # Get product info
        product_info = self.stock_data[self.stock_data['product_id'] == product_id].iloc[0]
        
        lead_time = 3  # Standard 3-day lead time for local suppliers
        
        # Get demand analysis
        demand_analysis = self.analyze_demand_patterns(product_id)
        daily_demand = demand_analysis['average_daily_demand']
        volatility = demand_analysis['volatility']
        
        # Calculate safety stock (for 95% service level)
        safety_stock = 1.65 * volatility * np.sqrt(lead_time)
        
        # Calculate reorder point
        reorder_point = (daily_demand * lead_time) + safety_stock
        
        # Determine urgency
        current_stock = product_info['current_stock']
        if current_stock <= reorder_point * 0.5:
            urgency = "CRITICAL"
        elif current_stock <= reorder_point:
            urgency = "HIGH"
        elif current_stock <= reorder_point * 1.5:
            urgency = "MEDIUM"
        else:
            urgency = "LOW"
        
        # Handle NaN values
        if np.isnan(reorder_point) or reorder_point <= 0:
            reorder_point = product_info['reorder_point']  # Fall back to existing reorder point
        
        return int(reorder_point), urgency
    
    def calculate_optimal_order_quantity(self, product_id: str) -> int:
        """Calculate optimal order quantity using Economic Order Quantity (EOQ) model"""
        product_info = self.stock_data[self.stock_data['product_id'] == product_id].iloc[0]
        
        # Get demand analysis
        demand_analysis = self.analyze_demand_patterns(product_id)
        annual_demand = demand_analysis['average_daily_demand'] * 365
        
        # EOQ parameters
        holding_cost_rate = 0.20  # 20% of unit cost per year
        ordering_cost = 50  # Fixed cost per order
        unit_cost = product_info['unit_cost']
        
        # Calculate EOQ
        if annual_demand > 0:
            eoq = np.sqrt((2 * annual_demand * ordering_cost) / (unit_cost * holding_cost_rate))
        else:
            eoq = product_info['max_stock'] - product_info['current_stock']
        
        min_order = 10  # Default minimum order quantity
        
        optimal_quantity = max(int(eoq), min_order)
        
        # Don't exceed maximum stock capacity
        max_order = product_info['max_stock'] - product_info['current_stock']
        optimal_quantity = min(optimal_quantity, max_order)
        
        return max(0, optimal_quantity)
    
    def create_inventory_plan(self) -> List[InventoryPlan]:
        """Create comprehensive inventory plan for all products"""
        if self.sales_data is None:
            self.load_data()
        
        plans = []
        
        for _, product in self.stock_data.iterrows():
            product_id = product['product_id']
            
            try:
                # Analyze current situation
                reorder_point, urgency = self.calculate_optimal_reorder_point(product_id)
                forecasted_demand = self.forecast_demand(product_id, 30)
                optimal_quantity = self.calculate_optimal_order_quantity(product_id)
                
                # Determine if reorder is needed
                current_stock = product['current_stock']
                
                # Calculate confidence based on historical data availability and quality
                product_sales_count = len(self.sales_data[self.sales_data['product_id'] == product_id])
                
                # Improved confidence calculation
                if product_sales_count >= 50:  # Good historical data (3+ months)
                    base_confidence = 0.85
                elif product_sales_count >= 20:  # Moderate data
                    base_confidence = 0.75
                elif product_sales_count >= 10:  # Limited data
                    base_confidence = 0.65
                else:  # Very limited data
                    base_confidence = 0.45
                
                # Add bonus for recent data (sales in last 30 days)
                recent_sales = self.sales_data[
                    (self.sales_data['product_id'] == product_id) & 
                    (self.sales_data['date'] >= datetime.now() - timedelta(days=30))
                ]
                if len(recent_sales) > 0:
                    base_confidence += 0.1
                
                # Add bonus for consistent demand patterns
                if product_sales_count > 10:
                    daily_sales = self.sales_data[self.sales_data['product_id'] == product_id].groupby('date')['quantity_sold'].sum()
                    if daily_sales.std() / (daily_sales.mean() + 1) < 0.5:  # Low coefficient of variation
                        base_confidence += 0.05
                
                confidence = min(0.95, base_confidence)
                
                # Determine reorder date based on current stock and demand
                if current_stock <= reorder_point:
                    reorder_date = datetime.now()
                else:
                    days_until_reorder = (current_stock - reorder_point) / max(1, forecasted_demand / 30)
                    reorder_date = datetime.now() + timedelta(days=max(0, days_until_reorder))
                
                # Create reasoning
                reasoning = f"Current stock: {current_stock}, Reorder point: {reorder_point}, "
                reasoning += f"30-day forecast: {forecasted_demand:.1f}, Recommended quantity: {optimal_quantity}"
                
                plan = InventoryPlan(
                    product_id=product_id,
                    product_name=product['product_name'],
                    current_stock=current_stock,
                    predicted_demand=forecasted_demand,
                    reorder_quantity=optimal_quantity,
                    reorder_date=reorder_date,
                    urgency_level=urgency,
                    reasoning=reasoning,
                    confidence_score=confidence
                )
                
                plans.append(plan)
                
            except Exception as e:
                self.logger.error(f"Error creating plan for {product_id}: {e}")
                continue
        
        # Sort by urgency and reorder date
        urgency_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        plans.sort(key=lambda x: (urgency_order.get(x.urgency_level, 4), x.reorder_date))
        
        self.logger.info(f"Created inventory plans for {len(plans)} products")
        return plans
    
    def process(self, action: str = "create_plan", **kwargs) -> Any:
        """Main processing method for the planner agent"""
        if action == "create_plan":
            return self.create_inventory_plan()
        elif action == "forecast_demand":
            product_id = kwargs.get('product_id')
            days_ahead = kwargs.get('days_ahead', 30)
            return self.forecast_demand(product_id, days_ahead)
        elif action == "analyze_patterns":
            product_id = kwargs.get('product_id')
            return self.analyze_demand_patterns(product_id)
        else:
            raise ValueError(f"Unknown action: {action}")

if __name__ == "__main__":
    # Test the planner agent
    planner = PlannerAgent()
    plans = planner.create_inventory_plan()
    
    print(f"\nGenerated {len(plans)} inventory plans:")
    for plan in plans[:5]:  # Show first 5 plans
        print(f"\nProduct: {plan.product_name}")
        print(f"Urgency: {plan.urgency_level}")
        print(f"Current Stock: {plan.current_stock}")
        print(f"Reorder Quantity: {plan.reorder_quantity}")
        print(f"Reasoning: {plan.reasoning}")
