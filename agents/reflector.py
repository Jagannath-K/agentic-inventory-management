"""
Reflector Agent - Performance analysis and system optimization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import logging
import json
import os
from .planner import BaseAgent, InventoryPlan

@dataclass
class PerformanceMetrics:
    """Data class for system performance metrics"""
    metric_name: str
    current_value: float
    target_value: float
    trend: str
    improvement_suggestion: str
    priority: str

@dataclass
class SystemInsight:
    """Data class for system insights and recommendations"""
    insight_type: str
    title: str
    description: str
    impact_level: str
    recommended_actions: List[str]
    estimated_benefit: str

class ReflectorAgent(BaseAgent):
    """
    AI Agent responsible for system performance analysis and optimization recommendations
    """
    
    def __init__(self):
        super().__init__("ReflectorAgent")
        self.sales_data = None
        self.stock_data = None
        self.order_log = None
        self.performance_history = []
        
    def load_data(self) -> None:
        """Load all necessary data for analysis"""
        try:
            self.sales_data = pd.read_csv('data/sales.csv')
            self.stock_data = pd.read_csv('data/stock.csv')
            
            # Convert date columns
            self.sales_data['date'] = pd.to_datetime(self.sales_data['date'])
            self.stock_data['last_updated'] = pd.to_datetime(self.stock_data['last_updated'])
            
            self.order_log = []
            self.logger.info("Order log loading disabled - using empty list for compatibility")
            
            self.logger.info("Reflector data loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            # Don't raise the exception, just log it and continue with minimal data
            if not hasattr(self, 'sales_data') or self.sales_data is None:
                self.sales_data = pd.DataFrame()
            if not hasattr(self, 'stock_data') or self.stock_data is None:
                self.stock_data = pd.DataFrame()
            if not hasattr(self, 'order_log') or self.order_log is None:
                self.order_log = []
    
    def calculate_inventory_turnover(self) -> Dict[str, float]:
        """Calculate inventory turnover ratios for all products"""
        if self.sales_data is None:
            self.load_data()
        
        turnover_ratios = {}
        
        for _, product in self.stock_data.iterrows():
            product_id = product['product_id']
            
            # Calculate total sales in the period
            product_sales = self.sales_data[self.sales_data['product_id'] == product_id]
            total_sold = product_sales['quantity_sold'].sum()
            
            # Calculate COGS (Cost of Goods Sold)
            cogs = total_sold * product['unit_cost']
            
            # Average inventory (simplified - using current stock)
            avg_inventory = product['current_stock'] * product['unit_cost']
            
            # Calculate turnover ratio
            if avg_inventory > 0:
                turnover_ratio = cogs / avg_inventory
            else:
                turnover_ratio = 0
            
            turnover_ratios[product_id] = turnover_ratio
        
        return turnover_ratios
    
    def analyze_stockout_risk(self) -> Dict[str, Dict[str, Any]]:
        """Analyze stockout risk for all products"""
        if self.sales_data is None:
            self.load_data()
        
        stockout_analysis = {}
        
        for _, product in self.stock_data.iterrows():
            product_id = product['product_id']
            current_stock = product['current_stock']
            reorder_point = product['reorder_point']
            
            # Calculate recent sales velocity
            recent_sales = self.sales_data[
                (self.sales_data['product_id'] == product_id) &
                (self.sales_data['date'] >= datetime.now() - timedelta(days=7))
            ]['quantity_sold'].sum()
            
            daily_velocity = recent_sales / 7 if recent_sales > 0 else 0
            
            # Calculate days until stockout
            if daily_velocity > 0:
                days_until_stockout = current_stock / daily_velocity
            else:
                days_until_stockout = float('inf')
            
            # Risk assessment
            if days_until_stockout <= 3:
                risk_level = "CRITICAL"
            elif days_until_stockout <= 7:
                risk_level = "HIGH"
            elif days_until_stockout <= 14:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            stockout_analysis[product_id] = {
                'current_stock': current_stock,
                'daily_velocity': daily_velocity,
                'days_until_stockout': days_until_stockout,
                'risk_level': risk_level,
                'below_reorder_point': current_stock < reorder_point
            }
        
        return stockout_analysis
    

    
    def calculate_key_performance_indicators(self) -> List[PerformanceMetrics]:
        """Calculate key performance indicators for the inventory system"""
        if self.sales_data is None:
            self.load_data()
        
        kpis = []
        
        # 1. Overall Inventory Turnover
        turnover_ratios = self.calculate_inventory_turnover()
        avg_turnover = np.mean(list(turnover_ratios.values())) if turnover_ratios else 0
        
        kpis.append(PerformanceMetrics(
            metric_name="Average Inventory Turnover",
            current_value=avg_turnover,
            target_value=6.0,  # Target: 6 times per year
            trend="stable" if 5.0 <= avg_turnover <= 7.0 else "needs_improvement",
            improvement_suggestion="Optimize reorder quantities and improve demand forecasting" if avg_turnover < 5.0 else "Good performance",
            priority="HIGH" if avg_turnover < 4.0 else "MEDIUM"
        ))
        
        # 2. Stockout Risk Assessment
        stockout_analysis = self.analyze_stockout_risk()
        high_risk_products = len([p for p in stockout_analysis.values() if p['risk_level'] in ['CRITICAL', 'HIGH']])
        total_products = len(stockout_analysis)
        stockout_risk_percentage = (high_risk_products / total_products * 100) if total_products > 0 else 0
        
        kpis.append(PerformanceMetrics(
            metric_name="High Stockout Risk Products (%)",
            current_value=stockout_risk_percentage,
            target_value=5.0,  # Target: <5% of products at high risk
            trend="critical" if stockout_risk_percentage > 15 else "good",
            improvement_suggestion="Implement more aggressive reordering for high-risk products" if stockout_risk_percentage > 5 else "Good risk management",
            priority="CRITICAL" if stockout_risk_percentage > 15 else "LOW"
        ))
        
        # 3. Inventory Value Efficiency
        total_inventory_value = sum(
            row['current_stock'] * row['unit_cost'] 
            for _, row in self.stock_data.iterrows()
        )
        
        # Calculate monthly sales value
        monthly_sales = self.sales_data[
            self.sales_data['date'] >= datetime.now() - timedelta(days=30)
        ]
        monthly_sales_value = (monthly_sales['quantity_sold'] * monthly_sales['unit_price']).sum()
        
        inventory_to_sales_ratio = (total_inventory_value / monthly_sales_value) if monthly_sales_value > 0 else 0
        
        kpis.append(PerformanceMetrics(
            metric_name="Inventory to Monthly Sales Ratio",
            current_value=inventory_to_sales_ratio,
            target_value=2.0,  # Target: 2 months of inventory
            trend="optimal" if 1.5 <= inventory_to_sales_ratio <= 2.5 else "needs_adjustment",
            improvement_suggestion="Reduce excess inventory" if inventory_to_sales_ratio > 3 else "Increase safety stock" if inventory_to_sales_ratio < 1 else "Well balanced",
            priority="MEDIUM" if abs(inventory_to_sales_ratio - 2.0) > 1.0 else "LOW"
        ))
        
        return kpis
    
    def generate_system_insights(self) -> List[SystemInsight]:
        """Generate actionable insights for system optimization"""
        insights = []
        
        # Analyze turnover patterns
        turnover_ratios = self.calculate_inventory_turnover()
        low_turnover_products = [
            pid for pid, ratio in turnover_ratios.items() if ratio < 2.0
        ]
        
        if low_turnover_products:
            insights.append(SystemInsight(
                insight_type="INVENTORY_OPTIMIZATION",
                title="Low Inventory Turnover Detected",
                description=f"{len(low_turnover_products)} products have low inventory turnover (< 2.0). This ties up capital and increases holding costs.",
                impact_level="MEDIUM",
                recommended_actions=[
                    "Review demand forecasting models for these products",
                    "Consider reducing reorder quantities",
                    "Implement promotional strategies to increase sales",
                    "Evaluate product lifecycle and consider discontinuation for consistently low performers"
                ],
                estimated_benefit="10-15% reduction in holding costs"
            ))
        
        # Analyze stockout risks
        stockout_analysis = self.analyze_stockout_risk()
        critical_products = [
            pid for pid, analysis in stockout_analysis.items() 
            if analysis['risk_level'] == 'CRITICAL'
        ]
        
        if critical_products:
            insights.append(SystemInsight(
                insight_type="SUPPLY_CHAIN_RISK",
                title="Critical Stockout Risk Identified",
                description=f"{len(critical_products)} products are at critical risk of stockout within 3 days.",
                impact_level="HIGH",
                recommended_actions=[
                    "Place emergency orders for critical products immediately",
                    "Review and adjust reorder points",
                    "Implement automated alerts for low stock situations",
                    "Consider expedited shipping for urgent orders"
                ],
                estimated_benefit="Prevent revenue loss and customer dissatisfaction"
            ))
        
        # Analyze seasonal patterns insight
        if len(self.sales_data) > 30:
            insights.append(SystemInsight(
                insight_type="DEMAND_FORECASTING",
                title="Implement Seasonal Demand Forecasting",
                description="Historical data suggests seasonal patterns that could improve demand prediction accuracy.",
                impact_level="MEDIUM",
                recommended_actions=[
                    "Implement time-series forecasting models",
                    "Adjust safety stock levels seasonally",
                    "Plan promotional campaigns around demand patterns",
                    "Coordinate with suppliers for seasonal capacity planning"
                ],
                estimated_benefit="15-20% improvement in forecast accuracy"
            ))
        
        return insights
    
    def create_optimization_report(self) -> Dict[str, Any]:
        """Create comprehensive optimization report with robust error handling"""
        try:
            if self.sales_data is None or len(self.sales_data) == 0:
                self.load_data()
            
            # Ensure we have minimal data to work with
            if self.sales_data is None or len(self.sales_data) == 0:
                return self._create_minimal_report("No sales data available")
            
            if self.stock_data is None or len(self.stock_data) == 0:
                return self._create_minimal_report("No stock data available")
            
            kpis = self.calculate_key_performance_indicators()
            insights = self.generate_system_insights()
            stockout_analysis = self.analyze_stockout_risk()
            
            # Prioritize recommendations
            high_priority_kpis = [kpi for kpi in kpis if kpi.priority == "CRITICAL"]
            high_impact_insights = [insight for insight in insights if hasattr(insight, 'impact_level') and insight.impact_level == "HIGH"]
            
            report = {
                'report_date': datetime.now().isoformat(),
                'summary': {
                    'total_products_analyzed': len(self.stock_data),
                    'high_priority_issues': len(high_priority_kpis) + len(high_impact_insights),
                    'overall_system_health': self._calculate_system_health_score(kpis)
                },
                'key_performance_indicators': [
                    {
                        'metric': kpi.metric_name,
                        'current': kpi.current_value,
                        'target': kpi.target_value,
                        'trend': kpi.trend,
                        'priority': kpi.priority,
                        'suggestion': kpi.improvement_suggestion
                    } for kpi in kpis
                ],
                'critical_insights': [
                    {
                        'type': insight.insight_type,
                        'title': insight.title,
                        'description': insight.description,
                        'impact': insight.impact_level,
                        'actions': insight.recommended_actions,
                        'benefit': insight.estimated_benefit
                    } for insight in insights
                ],
                'stockout_risks': {
                    pid: analysis for pid, analysis in stockout_analysis.items()
                    if analysis['risk_level'] in ['CRITICAL', 'HIGH']
                },
                'recommendations': self._generate_prioritized_recommendations(kpis, insights)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error creating optimization report: {e}")
            return self._create_minimal_report(f"Error generating report: {str(e)}")
    
    def _create_minimal_report(self, error_message: str) -> Dict[str, Any]:
        """Create a minimal report when full analysis fails"""
        return {
            'report_date': datetime.now().isoformat(),
            'summary': {
                'total_products_analyzed': 0,
                'high_priority_issues': 0,
                'overall_system_health': 'UNKNOWN'
            },
            'key_performance_indicators': [],
            'critical_insights': [{
                'type': 'SYSTEM_ERROR',
                'title': 'Report Generation Error',
                'description': error_message,
                'impact': 'HIGH',
                'actions': ['Check system logs', 'Verify data files', 'Restart the system'],
                'benefit': 'Restored system functionality'
            }],
            'stockout_risks': {},
            'recommendations': ['Check system configuration', 'Verify data integrity', 'Contact system administrator'],
            'error': error_message
        }
    
    def _calculate_system_health_score(self, kpis: List[PerformanceMetrics]) -> str:
        """Calculate overall system health score"""
        critical_issues = len([kpi for kpi in kpis if kpi.priority == "CRITICAL"])
        high_issues = len([kpi for kpi in kpis if kpi.priority == "HIGH"])
        
        if critical_issues > 0:
            return "CRITICAL"
        elif high_issues > 1:
            return "NEEDS_ATTENTION"
        elif high_issues == 1:
            return "GOOD"
        else:
            return "EXCELLENT"
    
    def _generate_prioritized_recommendations(self, kpis: List[PerformanceMetrics], insights: List[SystemInsight]) -> List[str]:
        """Generate specific, data-driven recommendations based on actual inventory conditions"""
        recommendations = []
        
        # Add critical KPI recommendations first
        for kpi in kpis:
            if kpi.priority == "CRITICAL":
                recommendations.append(f"🚨 URGENT: {kpi.improvement_suggestion}")
        
        # Add high-impact insights
        for insight in insights:
            if hasattr(insight, 'impact_level') and insight.impact_level == "HIGH":
                recommendations.append(f"⚡ HIGH IMPACT: {insight.recommended_actions[0]}")
        
        # Add other high-priority recommendations
        for kpi in kpis:
            if kpi.priority == "HIGH":
                recommendations.append(f"📊 Important: {kpi.improvement_suggestion}")
        
        # Generate specific recommendations based on current data analysis
        specific_recommendations = self._generate_specific_recommendations()
        recommendations.extend(specific_recommendations)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:6]  # Limit to top 6 recommendations
    
    def _generate_specific_recommendations(self) -> List[str]:
        """Generate specific, actionable recommendations based on current inventory data"""
        recommendations = []
        
        if self.stock_data is None or self.sales_data is None:
            return recommendations
        
        # 1. Analyze products with low stock relative to demand
        for _, product in self.stock_data.iterrows():
            product_id = product['product_id']
            product_name = product['product_name']
            current_stock = product['current_stock']
            reorder_point = product['reorder_point']
            max_stock = product['max_stock']
            
            # Calculate recent demand velocity
            recent_sales = self.sales_data[
                (self.sales_data['product_id'] == product_id) &
                (self.sales_data['date'] >= datetime.now() - timedelta(days=14))
            ]['quantity_sold'].sum()
            
            daily_velocity = recent_sales / 14 if recent_sales > 0 else 0
            days_of_stock = current_stock / daily_velocity if daily_velocity > 0 else float('inf')
            
            # Specific low stock recommendations
            if current_stock <= reorder_point and daily_velocity > 0:
                urgency = "within 3 days" if days_of_stock <= 3 else f"in {days_of_stock:.0f} days"
                recommendations.append(
                    f"� Reorder {product_name} immediately - current stock ({current_stock}) will run out {urgency} at current sales pace"
                )
            
            # Overstocking recommendations
            elif current_stock > max_stock * 0.8 and daily_velocity > 0 and days_of_stock > 30:
                recommendations.append(
                    f"📉 Reduce orders for {product_name} - overstocked with {days_of_stock:.0f} days of inventory (target: 15-20 days)"
                )
        
        # 2. Seasonal and category-specific recommendations
        current_month = datetime.now().month
        
        # Festival season recommendations (September-November in India)
        # Focus on items actually used during festivals
        if current_month in [9, 10, 11]:
            # Items that genuinely increase during festivals: staples for cooking, dairy for sweets, oil for frying
            festival_products = self.stock_data[
                self.stock_data['category'].isin(['Staples', 'Dairy']) |
                self.stock_data['product_name'].str.contains('Oil|Ghee|Sugar', case=False, na=False)
            ]
            
            for _, product in festival_products.iterrows():
                product_name = product['product_name']
                current_stock = product['current_stock']
                max_stock = product['max_stock']
                
                # Only recommend if significantly under capacity and it's a logical festival item
                if current_stock < max_stock * 0.5:  # Less than 50% of max capacity
                    if any(item in product_name.lower() for item in ['rice', 'flour', 'sugar', 'oil', 'ghee', 'milk']):
                        recommendations.append(
                            f"🎆 Increase {product_name} stock for festival season - demand rises 25-40% during Sept-Nov for cooking/sweets"
                        )
        
        # 3. High turnover products recommendations
        turnover_ratios = self.calculate_inventory_turnover()
        for product_id, turnover in turnover_ratios.items():
            if turnover > 12:  # Very high turnover (monthly+)
                product_info = self.stock_data[self.stock_data['product_id'] == product_id].iloc[0]
                product_name = product_info['product_name']
                current_stock = product_info['current_stock']
                reorder_point = product_info['reorder_point']
                
                if current_stock < reorder_point * 1.5:
                    recommendations.append(
                        f"⚡ {product_name} is fast-moving (turnover: {turnover:.1f}x/year) - consider increasing safety stock by 25% to prevent stockouts"
                    )
        
        # 4. Slow-moving products recommendations
        for product_id, turnover in turnover_ratios.items():
            if turnover < 2:  # Very slow turnover
                product_info = self.stock_data[self.stock_data['product_id'] == product_id].iloc[0]
                product_name = product_info['product_name']
                current_stock = product_info['current_stock']
                unit_cost = product_info['unit_cost']
                tied_capital = current_stock * unit_cost
                
                if tied_capital > 1000:  # Significant capital tied up
                    recommendations.append(
                        f"� {product_name} has slow turnover ({turnover:.1f}x/year) with ₹{tied_capital:.0f} tied up - consider promotional pricing or smaller orders"
                    )
        
        # 5. Cost optimization recommendations
        high_value_products = self.stock_data[self.stock_data['unit_cost'] > 200]
        for _, product in high_value_products.iterrows():
            product_name = product['product_name']
            current_stock = product['current_stock']
            unit_cost = product['unit_cost']
            inventory_value = current_stock * unit_cost
            
            # Get recent sales for this high-value product
            recent_sales = self.sales_data[
                (self.sales_data['product_id'] == product['product_id']) &
                (self.sales_data['date'] >= datetime.now() - timedelta(days=30))
            ]['quantity_sold'].sum()
            
            if inventory_value > 5000 and recent_sales < 10:  # High value, low movement
                recommendations.append(
                    f"💎 {product_name} has high inventory value (₹{inventory_value:.0f}) but low sales ({recent_sales} units/month) - optimize order frequency"
                )
        
        # 6. Category-specific recommendations based on perishability
        fresh_categories = ['Vegetables', 'Dairy', 'Bakery']
        fresh_products = self.stock_data[self.stock_data['category'].isin(fresh_categories)]
        
        for _, product in fresh_products.iterrows():
            product_name = product['product_name']
            current_stock = product['current_stock']
            
            # Calculate days of stock for fresh items
            recent_sales = self.sales_data[
                (self.sales_data['product_id'] == product['product_id']) &
                (self.sales_data['date'] >= datetime.now() - timedelta(days=7))
            ]['quantity_sold'].sum()
            
            daily_velocity = recent_sales / 7 if recent_sales > 0 else 0
            days_of_stock = current_stock / daily_velocity if daily_velocity > 0 else float('inf')
            
            if days_of_stock > 7:  # More than a week of fresh products
                recommendations.append(
                    f"🥬 {product_name} (fresh category) has {days_of_stock:.0f} days of stock - consider reducing orders to minimize spoilage"
                )
        
        return recommendations
    
    def process(self, action: str = "create_report", **kwargs) -> Any:
        """Main processing method for the reflector agent"""
        if action == "create_report":
            return self.create_optimization_report()
        elif action == "analyze_kpis":
            return self.calculate_key_performance_indicators()
        elif action == "generate_insights":
            return self.generate_system_insights()
        elif action == "analyze_stockouts":
            return self.analyze_stockout_risk()
        else:
            raise ValueError(f"Unknown action: {action}")

if __name__ == "__main__":
    # Test the reflector agent
    reflector = ReflectorAgent()
    report = reflector.create_optimization_report()
    
    print("System Optimization Report")
    print("=" * 50)
    print(f"Report Date: {report['report_date']}")
    print(f"Overall System Health: {report['summary']['overall_system_health']}")
    print(f"High Priority Issues: {report['summary']['high_priority_issues']}")
    
    print("\nKey Performance Indicators:")
    for kpi in report['key_performance_indicators']:
        print(f"- {kpi['metric']}: {kpi['current']:.2f} (Target: {kpi['target']:.2f}) - {kpi['priority']}")
    
    print("\nTop Recommendations:")
    for i, rec in enumerate(report['recommendations'][:5], 1):
        print(f"{i}. {rec}")
