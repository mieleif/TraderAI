"""
TraderAI Strategies Package

This package contains all the strategy-related modules for the TraderAI system:

- Ichimoku Signals: Generate trading signals based on Ichimoku Cloud indicators
- Feature Engineering: Create ML-friendly features from price data and indicators
- Hybrid Decision System: Combine AI predictions with technical signals
- Backtest Engine: Comprehensive backtesting with performance metrics
- Visualization: Advanced trading visualization tools
"""

from .ichimoku_signals import IchimokuSignals
from .feature_engineering import FeatureEngineering
from .hybrid_decision import HybridDecisionSystem
from .backtest_engine import BacktestEngine
from .visualization import TradingVisualization

__all__ = [
    'IchimokuSignals',
    'FeatureEngineering', 
    'HybridDecisionSystem',
    'BacktestEngine',
    'TradingVisualization'
]