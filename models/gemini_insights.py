"""
Google Gemini Integration for Agentic Inventory System
Provides AI-powered insights and natural language capabilities
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

try:
    import google.generativeai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class GeminiInsightGenerator:
    """
    Generate AI-powered insights for inventory management using Google Gemini
    """
    
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Gemini model"""
        if not GEMINI_AVAILABLE:
            logger.warning("Google Gemini packages not installed. Install with: pip install google-generativeai langchain-google-genai")
            return
            
        if not self.api_key or self.api_key == "your_google_gemini_api_key_here":
            logger.warning("Google API key not set. Please set GOOGLE_API_KEY in your .env file")
            return
            
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            logger.info("Google Gemini model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {e}")
    
    def is_available(self) -> bool:
        """Check if Gemini is available and configured"""
        return self.model is not None
    
    def generate_inventory_insights(self, 
                                  stock_data: pd.DataFrame,
                                  sales_data: pd.DataFrame,
                                  performance_metrics: Dict[str, Any]) -> str:
        """Generate comprehensive inventory insights"""
        if not self.is_available():
            return "AI insights unavailable - Google Gemini not configured"
        
        # Prepare context for the AI
        context = self._prepare_analysis_context(stock_data, sales_data, performance_metrics)
        
        prompt = f"""
        As an expert inventory management consultant, analyze the following inventory data and provide actionable insights:

        {context}

        Please provide:
        1. **Key Findings**: Most important observations about inventory performance
        2. **Risk Assessment**: Products at risk of stockouts or overstocking
        3. **Optimization Opportunities**: Specific recommendations to improve efficiency
        4. **Strategic Recommendations**: Long-term suggestions for inventory management
        5. **Action Items**: Immediate steps to take

        Keep the analysis practical and focused on business value.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return f"Error generating AI insights: {e}"
    
    def analyze_product_performance(self, product_id: str, product_data: Dict[str, Any]) -> str:
        """Generate specific insights for a single product"""
        if not self.is_available():
            return "AI analysis unavailable"
        
        prompt = f"""
        Analyze this specific product's inventory performance:
        
        Product ID: {product_id}
        Current Stock: {product_data.get('current_stock', 'N/A')}
        Sales Velocity: {product_data.get('sales_velocity', 'N/A')} units/week
        Stock Level: {product_data.get('stock_level', 'N/A')}
        Days Since Last Order: {product_data.get('days_since_last_order', 'N/A')}
        
        Provide a concise analysis with:
        1. Current status assessment
        2. Risk level (Low/Medium/High)
        3. Specific recommendation
        4. Timeline for action
        
        Be specific and actionable.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error analyzing product {product_id}: {e}")
            return f"Analysis error: {e}"
    
    def explain_decision(self, decision_type: str, details: Dict[str, Any]) -> str:
        """Explain why a specific inventory decision was made"""
        if not self.is_available():
            return "Decision explanation unavailable"
        
        prompt = f"""
        Explain this inventory management decision in simple terms:
        
        Decision Type: {decision_type}
        Details: {details}
        
        Provide a clear, non-technical explanation that covers:
        1. What decision was made
        2. Why it was necessary
        3. Expected benefits
        4. What happens next
        
        Write for a business manager audience.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error explaining decision: {e}")
            return f"Explanation error: {e}"
    
    def _prepare_analysis_context(self, 
                                stock_data: pd.DataFrame,
                                sales_data: pd.DataFrame, 
                                metrics: Dict[str, Any]) -> str:
        """Prepare context data for AI analysis"""
        
        # Stock summary
        total_products = len(stock_data)
        low_stock_count = len(stock_data[stock_data['current_stock'] < stock_data['reorder_point']])
        total_inventory_value = (stock_data['current_stock'] * stock_data['unit_cost']).sum()
        
        # Sales summary
        recent_sales = sales_data[sales_data['date'] >= (datetime.now() - pd.Timedelta(days=30))]
        monthly_revenue = (recent_sales['quantity_sold'] * recent_sales['price']).sum()
        
        context = f"""
        INVENTORY OVERVIEW:
        - Total Products: {total_products}
        - Products Below Reorder Point: {low_stock_count}
        - Total Inventory Value: ${total_inventory_value:,.2f}
        - Monthly Revenue (Last 30 days): ${monthly_revenue:,.2f}
        
        KEY METRICS:
        """
        
        for metric_name, metric_value in metrics.items():
            context += f"- {metric_name}: {metric_value}\n"
        
        # Top 5 products by stock level
        top_stock = stock_data.nlargest(5, 'current_stock')[['product_name', 'current_stock', 'reorder_point']]
        context += f"\nTOP 5 PRODUCTS BY STOCK:\n{top_stock.to_string()}\n"
        
        # Low stock products
        low_stock = stock_data[stock_data['current_stock'] < stock_data['reorder_point']]
        if not low_stock.empty:
            context += f"\nLOW STOCK PRODUCTS:\n{low_stock[['product_name', 'current_stock', 'reorder_point']].to_string()}\n"
        
        return context

# Global instance
gemini_insights = GeminiInsightGenerator()
