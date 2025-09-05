# Models module initialization
from .predictor import DemandPredictor, SeasonalityAnalyzer
from .shop_operations import ShopOperations
from .notification_system import NotificationSystem

try:
    from .gemini_insights import GeminiInsightGenerator, gemini_insights
    __all__ = ['DemandPredictor', 'SeasonalityAnalyzer', 'ShopOperations', 'NotificationSystem', 'GeminiInsightGenerator', 'gemini_insights']
except ImportError:
    # Gemini dependencies not installed
    __all__ = ['DemandPredictor', 'SeasonalityAnalyzer', 'ShopOperations', 'NotificationSystem']
