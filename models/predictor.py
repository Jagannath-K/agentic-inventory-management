"""
Demand Prediction Models - Machine Learning models for inventory forecasting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DemandPredictor:
    """
    Advanced demand prediction using multiple machine learning models
    Specialized for grocery retail with Indian market pattern recognition
    """
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'linear_regression': LinearRegression()
        }
        self.scalers = {}
        self.trained_models = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.sales_data = None
        self.stock_data = None
        self.reasoning_patterns = {
            'month_start': "Higher demand expected due to salary payments (1st-5th of month)",
            'month_mid': "Moderate demand expected during mid-month period",
            'month_end': "Lower demand expected as customers await next salary",
            'weekend': "Increased demand due to weekend shopping patterns",
            'weekday': "Normal weekday demand pattern",
            'staples_high': "Staple products show consistent high demand",
            'dairy_daily': "Daily consumption items maintain steady demand",
            'seasonal_spike': "Festival/seasonal demand increase expected"
        }
        
    def load_data(self) -> None:
        """Load and prepare data for modeling"""
        try:
            self.sales_data = pd.read_csv('data/sales.csv')
            self.stock_data = pd.read_csv('data/stock.csv')
            
            self.sales_data['date'] = pd.to_datetime(self.sales_data['date'], errors='coerce')
            
            logger.info("Data loaded successfully for demand prediction")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_features(self, product_id: str, target_date: datetime = None) -> pd.DataFrame:
        """Create feature matrix for demand prediction"""
        if self.sales_data is None:
            self.load_data()
        
        if target_date is None:
            target_date = datetime.now()
        
        # Filter sales data for the product
        product_sales = self.sales_data[self.sales_data['product_id'] == product_id].copy()
        
        if product_sales.empty:
            # Return default features for new products
            return pd.DataFrame({
                'day_of_week': [target_date.weekday()],
                'month': [target_date.month],
                'day_of_month': [target_date.day],
                'quarter': [target_date.quarter],
                'is_weekend': [1 if target_date.weekday() >= 5 else 0],
                'days_since_launch': [1],
                'avg_demand_7d': [0],
                'avg_demand_14d': [0],
                'avg_demand_30d': [0],
                'trend_7d': [0],
                'volatility_7d': [0],
                'price_trend': [0],
                'seasonal_factor': [1.0]
            })
        
        # Create daily aggregated data (micro enterprises focus on retail only)
        daily_sales = product_sales.groupby('date').agg({
            'quantity_sold': 'sum',
            'unit_price': 'mean'
        }).reset_index()
        
        # Sort by date
        daily_sales = daily_sales.sort_values('date')
        
        # Create feature matrix
        features_list = []
        
        # Generate features for each date in the sales history
        for i, row in daily_sales.iterrows():
            current_date = row['date']
            
            # Time-based features
            features = {
                'day_of_week': current_date.weekday(),
                'month': current_date.month,
                'day_of_month': current_date.day,
                'quarter': current_date.quarter,
                'is_weekend': 1 if current_date.weekday() >= 5 else 0,
                'days_since_launch': (current_date - daily_sales['date'].min()).days + 1
            }
            
            # Historical demand features
            recent_sales = daily_sales[daily_sales['date'] < current_date]['quantity_sold']
            
            if len(recent_sales) >= 7:
                features['avg_demand_7d'] = recent_sales.tail(7).mean()
                features['trend_7d'] = recent_sales.tail(7).mean() - recent_sales.tail(14).head(7).mean()
                features['volatility_7d'] = recent_sales.tail(7).std()
            else:
                features['avg_demand_7d'] = recent_sales.mean() if len(recent_sales) > 0 else 0
                features['trend_7d'] = 0
                features['volatility_7d'] = 0
            
            if len(recent_sales) >= 14:
                features['avg_demand_14d'] = recent_sales.tail(14).mean()
            else:
                features['avg_demand_14d'] = features['avg_demand_7d']
            
            if len(recent_sales) >= 30:
                features['avg_demand_30d'] = recent_sales.tail(30).mean()
            else:
                features['avg_demand_30d'] = features['avg_demand_14d']
            
            # Price trend features
            recent_prices = daily_sales[daily_sales['date'] <= current_date]['unit_price']
            if len(recent_prices) >= 2:
                features['price_trend'] = recent_prices.iloc[-1] - recent_prices.iloc[-2]
            else:
                features['price_trend'] = 0
            
            # Seasonal factors (simplified - based on month for Indian market)
            seasonal_multipliers = {
                1: 0.9, 2: 0.8, 3: 0.9, 4: 1.0, 5: 1.1, 6: 1.2,
                7: 1.3, 8: 1.2, 9: 1.0, 10: 1.1, 11: 1.3, 12: 1.4
            }
            features['seasonal_factor'] = seasonal_multipliers.get(current_date.month, 1.0)
            
            # Target variable
            features['target'] = row['quantity_sold']
            features['date'] = current_date
            
            features_list.append(features)
        
        # Create DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Handle missing values
        features_df = features_df.fillna(0)
        
        return features_df
    
    def create_prediction_features(self, product_id: str, target_date: datetime) -> pd.DataFrame:
        """Create features specifically for prediction on a target date"""
        if self.sales_data is None:
            self.load_data()
        
        # Filter sales data for the product
        product_sales = self.sales_data[self.sales_data['product_id'] == product_id].copy()
        
        if product_sales.empty:
            # Return default features for new products with target date characteristics
            return pd.DataFrame({
                'day_of_week': [target_date.weekday()],
                'month': [target_date.month],
                'day_of_month': [target_date.day],
                'quarter': [(target_date.month - 1) // 3 + 1],  # Calculate quarter manually
                'is_weekend': [1 if target_date.weekday() >= 5 else 0],
                'days_since_launch': [1],
                'avg_demand_7d': [1],
                'avg_demand_14d': [1],
                'avg_demand_30d': [1],
                'trend_7d': [0],
                'volatility_7d': [0.5],
                'price_trend': [0],
                'seasonal_factor': [1.0]
            })
        
        # Create daily aggregated data
        daily_sales = product_sales.groupby('date').agg({
            'quantity_sold': 'sum',
            'unit_price': 'mean'
        }).reset_index()
        daily_sales = daily_sales.sort_values('date')
        
        # Create features for the target date using historical data
        features = {
            'day_of_week': target_date.weekday(),
            'month': target_date.month,
            'day_of_month': target_date.day,
            'quarter': (target_date.month - 1) // 3 + 1,  # Calculate quarter manually
            'is_weekend': 1 if target_date.weekday() >= 5 else 0,
            'days_since_launch': (target_date - daily_sales['date'].min()).days + 1
        }
        
        # Use all historical sales for features (since we're predicting future)
        all_sales = daily_sales['quantity_sold']
        
        if len(all_sales) >= 7:
            features['avg_demand_7d'] = all_sales.tail(7).mean()
            if len(all_sales) >= 14:
                features['trend_7d'] = all_sales.tail(7).mean() - all_sales.tail(14).head(7).mean()
            else:
                features['trend_7d'] = 0
            features['volatility_7d'] = all_sales.tail(7).std()
        else:
            features['avg_demand_7d'] = all_sales.mean() if len(all_sales) > 0 else 1
            features['trend_7d'] = 0
            features['volatility_7d'] = all_sales.std() if len(all_sales) > 1 else 0.5
        
        if len(all_sales) >= 14:
            features['avg_demand_14d'] = all_sales.tail(14).mean()
        else:
            features['avg_demand_14d'] = features['avg_demand_7d']
        
        if len(all_sales) >= 30:
            features['avg_demand_30d'] = all_sales.tail(30).mean()
        else:
            features['avg_demand_30d'] = features['avg_demand_14d']
        
        # Price trend features
        all_prices = daily_sales['unit_price']
        if len(all_prices) >= 2:
            features['price_trend'] = all_prices.iloc[-1] - all_prices.iloc[-2]
        else:
            features['price_trend'] = 0
        
        # Seasonal factors (simplified - based on month for Indian market)
        seasonal_multipliers = {
            1: 0.9, 2: 0.8, 3: 0.9, 4: 1.0, 5: 1.1, 6: 1.2,
            7: 1.3, 8: 1.2, 9: 1.0, 10: 1.1, 11: 1.3, 12: 1.4
        }
        features['seasonal_factor'] = seasonal_multipliers.get(target_date.month, 1.0)
        
        return pd.DataFrame([features])

    def prepare_training_data(self, product_id: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data for a specific product"""
        features_df = self.create_features(product_id)
        
        if len(features_df) < 5:  # Need at least 5 data points
            return None, None, None
        
        # Feature columns (exclude target and date)
        feature_columns = [col for col in features_df.columns if col not in ['target', 'date']]
        
        X = features_df[feature_columns].values
        y = features_df['target'].values
        
        return X, y, feature_columns
    
    def train_models(self, product_id: str) -> Dict[str, Any]:
        """Train all models for a specific product"""
        X, y, feature_columns = self.prepare_training_data(product_id)
        
        if X is None:
            logger.warning(f"Insufficient data for training models for product {product_id}")
            return {}
        
        # Split data for validation
        if len(X) >= 10:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
        
        trained_models = {}
        performance_results = {}
        
        for model_name, model in self.models.items():
            try:
                # Scale features for models that benefit from scaling
                if model_name in ['linear_regression']:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    self.scalers[f"{product_id}_{model_name}"] = scaler
                else:
                    X_train_scaled = X_train
                    X_test_scaled = X_test
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
                
                # Calculate performance metrics
                train_mae = mean_absolute_error(y_train, y_pred_train)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                
                performance_results[model_name] = {
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_r2': train_r2,
                    'test_r2': test_r2
                }
                
                # Store trained model
                trained_models[model_name] = model
                
                # Feature importance (for tree-based models)
                if hasattr(model, 'feature_importances_'):
                    importance_dict = dict(zip(feature_columns, model.feature_importances_))
                    self.feature_importance[f"{product_id}_{model_name}"] = importance_dict
                
                logger.info(f"Trained {model_name} for {product_id} - Test MAE: {test_mae:.2f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name} for {product_id}: {e}")
                continue
        
        # Store results
        self.trained_models[product_id] = trained_models
        self.model_performance[product_id] = performance_results
        
        return {
            'models_trained': list(trained_models.keys()),
            'performance': performance_results,
            'best_model': min(performance_results.keys(), 
                            key=lambda x: performance_results[x]['test_mae']) if performance_results else None
        }
    
    def predict_demand(self, product_id: str, target_date: datetime = None, days_ahead: int = 1) -> Dict[str, Any]:
        """Predict demand for a product on a specific date with intelligent reasoning"""
        if target_date is None:
            target_date = datetime.now() + timedelta(days=days_ahead)
        
        # Check if models are trained for this product
        if product_id not in self.trained_models or not self.trained_models[product_id]:
            # Train models if not available
            self.train_models(product_id)
        
        if product_id not in self.trained_models or not self.trained_models[product_id]:
            # Fallback to simple historical average
            if self.sales_data is None:
                self.load_data()
            
            product_sales = self.sales_data[self.sales_data['product_id'] == product_id]
            if not product_sales.empty:
                avg_demand = product_sales['quantity_sold'].mean()
            else:
                avg_demand = 1  # Default for new products
            
            reasoning = self._generate_reasoning(product_id, target_date, avg_demand, method='historical')
            
            return {
                'predicted_demand': avg_demand,
                'method': 'historical_average',
                'confidence': 0.5,
                'model_predictions': {},
                'ensemble_prediction': avg_demand,
                'reasoning': reasoning
            }
        
        # Create features for the target date using the new prediction method
        features_df = self.create_prediction_features(product_id, target_date)
        feature_columns = [col for col in features_df.columns if col not in ['target', 'date']]
        X_pred = features_df[feature_columns].values
        
        # Get predictions from all trained models
        model_predictions = {}
        for model_name, model in self.trained_models[product_id].items():
            try:
                # Apply scaling if needed
                if f"{product_id}_{model_name}" in self.scalers:
                    X_pred_scaled = self.scalers[f"{product_id}_{model_name}"].transform(X_pred)
                else:
                    X_pred_scaled = X_pred
                
                prediction = model.predict(X_pred_scaled)[0]
                model_predictions[model_name] = max(0, prediction)  # Ensure non-negative
                
            except Exception as e:
                logger.error(f"Error predicting with {model_name} for {product_id}: {e}")
                continue
        
        if not model_predictions:
            return self.predict_demand(product_id, target_date, days_ahead)  # Fallback
        
        # Ensemble prediction (weighted average based on performance)
        if product_id in self.model_performance:
            weights = {}
            for model_name in model_predictions.keys():
                if model_name in self.model_performance[product_id]:
                    # Weight based on inverse of test MAE (lower MAE = higher weight)
                    mae = self.model_performance[product_id][model_name]['test_mae']
                    weights[model_name] = 1 / (mae + 0.1)  # Add small constant to avoid division by zero
                else:
                    weights[model_name] = 1.0
            
            # Normalize weights
            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items()}
            
            # Calculate weighted average
            ensemble_prediction = sum(pred * weights[model_name] 
                                    for model_name, pred in model_predictions.items())
        else:
            # Simple average if no performance data
            ensemble_prediction = np.mean(list(model_predictions.values()))
        
        # Calculate confidence based on model agreement
        predictions_array = np.array(list(model_predictions.values()))
        prediction_std = np.std(predictions_array)
        mean_prediction = np.mean(predictions_array)
        
        # Confidence calculation (higher agreement = higher confidence)
        if mean_prediction > 0:
            coefficient_of_variation = prediction_std / mean_prediction
            confidence = max(0.3, min(0.95, 1 - coefficient_of_variation))
        else:
            confidence = 0.5
        
        # Generate intelligent reasoning
        reasoning = self._generate_reasoning(product_id, target_date, ensemble_prediction, method='ml')
        
        return {
            'predicted_demand': round(ensemble_prediction, 2),
            'method': 'ensemble_ml',
            'confidence': round(confidence, 3),
            'model_predictions': {k: round(v, 2) for k, v in model_predictions.items()},
            'ensemble_prediction': round(ensemble_prediction, 2),
            'reasoning': reasoning,
            'prediction_details': {
                'target_date': target_date.strftime('%Y-%m-%d'),
                'features_used': feature_columns,
                'models_used': list(model_predictions.keys())
            }
        }
    
    def _generate_reasoning(self, product_id: str, target_date: datetime, prediction: float, method: str = 'ml') -> str:
        """Generate intelligent reasoning for demand predictions based on Indian market patterns"""
        reasoning_parts = []
        
        # Get product info
        if self.stock_data is not None:
            product_info = self.stock_data[self.stock_data['product_id'] == product_id]
            if not product_info.empty:
                category = product_info.iloc[0]['category']
                product_name = product_info.iloc[0]['product_name']
            else:
                category = 'General'
                product_name = product_id
        else:
            category = 'General'
            product_name = product_id
        
        # Temporal patterns
        day_of_month = target_date.day
        day_of_week = target_date.weekday()  # 0=Monday, 6=Sunday
        month = target_date.month
        
        # Month-based reasoning (Indian salary patterns)
        if day_of_month <= 5:
            reasoning_parts.append("📅 Higher demand expected as it's the beginning of month (salary payment period)")
        elif 15 <= day_of_month <= 18:
            reasoning_parts.append("📅 Moderate increase expected during mid-month period")
        elif day_of_month >= 26:
            reasoning_parts.append("📅 Lower demand expected as customers typically have reduced spending at month-end")
        
        # Day of week patterns
        if day_of_week >= 5:  # Weekend
            reasoning_parts.append("🛒 Weekend shopping boost expected, especially for grocery items")
        elif day_of_week == 0:  # Monday
            reasoning_parts.append("📦 Monday restocking pattern observed in grocery retail")
        
        # Category-specific insights
        if category == 'Staples':
            reasoning_parts.append("🌾 Staple products maintain consistent demand with bulk purchase patterns")
        elif category == 'Dairy':
            reasoning_parts.append("🥛 Daily consumption item with regular customer purchase cycles")
        elif category == 'Snacks':
            reasoning_parts.append("🍪 Snack demand influenced by weekend and evening consumption patterns")
        elif category == 'Personal Care':
            reasoning_parts.append("🧼 Personal care items show steady demand with monthly replenishment cycles")
        elif category == 'Vegetables':
            reasoning_parts.append("🥬 Fresh vegetables require frequent restocking due to perishability")
        
        # Prediction magnitude reasoning
        if prediction >= 3:
            reasoning_parts.append(f"📈 High demand prediction ({prediction:.1f} units) suggests strong market need")
        elif prediction >= 1.5:
            reasoning_parts.append(f"📊 Moderate demand prediction ({prediction:.1f} units) indicates stable consumption")
        else:
            reasoning_parts.append(f"📉 Lower demand prediction ({prediction:.1f} units) suggests reduced consumption")
        
        # Method-specific confidence
        if method == 'ml':
            reasoning_parts.append("🤖 Prediction based on ensemble ML models analyzing historical patterns")
        else:
            reasoning_parts.append("📊 Prediction based on historical average demand analysis")
        
        # Seasonal considerations
        if month in [10, 11]:  # Festival season
            reasoning_parts.append("🎆 Festival season may increase demand for certain categories")
        elif month in [3, 4]:  # Summer season
            reasoning_parts.append("☀️ Summer season patterns considered in demand calculation")
        
        return " | ".join(reasoning_parts)
    
    def batch_predict(self, product_ids: List[str], days_ahead: int = 7) -> Dict[str, Dict[str, Any]]:
        """Predict demand for multiple products over multiple days"""
        predictions = {}
        
        for product_id in product_ids:
            product_predictions = {}
            
            for day in range(1, days_ahead + 1):
                target_date = datetime.now() + timedelta(days=day)
                pred_result = self.predict_demand(product_id, target_date)
                product_predictions[f"day_{day}"] = pred_result
            
            predictions[product_id] = product_predictions
        
        return predictions
    
    def save_models(self, filepath_prefix: str = "models/demand_predictor") -> None:
        """Save trained models to disk"""
        os.makedirs(os.path.dirname(filepath_prefix), exist_ok=True)
        
        # Save all components
        joblib.dump(self.trained_models, f"{filepath_prefix}_models.pkl")
        joblib.dump(self.scalers, f"{filepath_prefix}_scalers.pkl")
        joblib.dump(self.feature_importance, f"{filepath_prefix}_importance.pkl")
        joblib.dump(self.model_performance, f"{filepath_prefix}_performance.pkl")
        
        logger.info(f"Models saved to {filepath_prefix}")
    
    def load_models(self, filepath_prefix: str = "models/demand_predictor") -> None:
        """Load trained models from disk"""
        try:
            self.trained_models = joblib.load(f"{filepath_prefix}_models.pkl")
            self.scalers = joblib.load(f"{filepath_prefix}_scalers.pkl")
            self.feature_importance = joblib.load(f"{filepath_prefix}_importance.pkl")
            self.model_performance = joblib.load(f"{filepath_prefix}_performance.pkl")
            
            logger.info(f"Models loaded from {filepath_prefix}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def get_feature_importance(self, product_id: str, model_name: str = 'random_forest') -> Dict[str, float]:
        """Get feature importance for a specific product and model"""
        key = f"{product_id}_{model_name}"
        return self.feature_importance.get(key, {})
    
    def evaluate_model_performance(self, product_id: str) -> Dict[str, Any]:
        """Get comprehensive model performance evaluation"""
        if product_id not in self.model_performance:
            return {}
        
        performance = self.model_performance[product_id]
        
        # Find best model based on test MAE
        best_model = min(performance.keys(), key=lambda x: performance[x]['test_mae'])
        
        return {
            'best_model': best_model,
            'model_performance': performance,
            'feature_importance': self.get_feature_importance(product_id, best_model)
        }



if __name__ == "__main__":
    # Test the demand predictor
    predictor = DemandPredictor()
    
    # Train models for a sample product
    training_result = predictor.train_models('P001')
    print("Training Results:", training_result)
    
    # Make predictions
    prediction = predictor.predict_demand('P001', days_ahead=7)
    print(f"\nPrediction for P001 (7 days ahead): {prediction['predicted_demand']:.2f}")
    print(f"Confidence: {prediction['confidence']:.2f}")
    print(f"Method: {prediction['method']}")
    
    # Batch predictions
    batch_predictions = predictor.batch_predict(['P001', 'P002'], days_ahead=3)
    print(f"\nBatch predictions for 3 days:")
    for product, preds in batch_predictions.items():
        print(f"{product}: {[preds[f'day_{i}']['predicted_demand'] for i in range(1, 4)]}")
