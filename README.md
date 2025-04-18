# TraderAI: AI-Powered Cryptocurrency Trading System

A sophisticated trading system that combines reinforcement learning algorithms (PPO and DQN) with traditional Ichimoku Cloud trading signals to make optimized cryptocurrency trading decisions.

## Overview

TraderAI is a comprehensive trading system that:

1. Uses deep reinforcement learning to learn optimal trading strategies
2. Leverages traditional Ichimoku Cloud indicators for technical analysis
3. Combines AI predictions with technical signals in a hybrid decision system
4. Provides advanced backtesting and performance visualization
5. Features risk management and position sizing logic

## Project Structure

The project is organized into two main implementations:

1. **DQN** - Implementation using Deep Q-Networks (discrete action space)
2. **PPO** - Implementation using Proximal Policy Optimization (continuous action space)

### Key Components

- **Enhanced Trading Environment**: Custom OpenAI Gym environment for cryptocurrency trading simulation
- **Reinforcement Learning Agents**: DQN and PPO implementations for learning trading strategies
- **Ichimoku Signal Module**: Generates trading signals from Ichimoku Cloud indicators
- **Feature Engineering**: Creates ML-friendly features from price data and indicators
- **Hybrid Decision System**: Combines AI and Ichimoku signals for final trading decisions
- **Backtest Engine**: Comprehensive backtesting with realistic simulations and performance metrics
- **Visualization Tools**: Advanced visualization of trading signals, performance, and Ichimoku components

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/traderAI.git
   cd traderAI
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r PPO/requirements.txt
   ```

## Usage

The system can be used in multiple modes:

### Training Mode

Train a new PPO agent on historical cryptocurrency data:

```bash
cd PPO
python main.py --mode train --data data/ETHUSDT_4h_data.csv
```

### Backtesting Mode

Backtest the trained agent on historical data:

```bash
cd PPO
python main.py --mode backtest --data data/ETHUSDT_4h_data.csv --visualize
```

### Hybrid Strategy Mode

Run the hybrid strategy that combines AI predictions with Ichimoku signals:

```bash
cd PPO
python main.py --mode hybrid --data data/ETHUSDT_4h_data.csv --visualize
```

### Command-line Arguments

- `--mode`: Operation mode (`train`, `backtest`, or `hybrid`)
- `--data`: Path to the data file
- `--model`: Path to a pretrained model (for backtest and hybrid modes)
- `--visualize`: Generate visualizations
- `--sample_size`: Number of data points to use (use 0 for all)

## Components in Detail

### Ichimoku Signal Module

The `ichimoku_signals.py` module implements:

- TK Cross detection (Tenkan-Kijun crossover)
- Kumo breakout signals (price breaking above/below cloud)
- Kumo twist detection (when Senkou Span A crosses Senkou Span B)
- Price-Kijun relationship analysis
- Chikou Span confirmation
- Signal strength and confidence scoring

### Feature Engineering

The `feature_engineering.py` module:

- Extracts useful features from Ichimoku components 
- Normalizes data for ML models
- Calculates technical indicators
- Creates time series features
- Handles feature scaling and preparation

### Hybrid Decision System

The `hybrid_decision.py` module:

- Weighs AI and Ichimoku signals
- Calculates combined signals and confidence
- Determines position sizes
- Implements risk management
- Makes final trading decisions
- Provides performance analysis

### Backtest Engine

The `backtest_engine.py` module:

- Simulates trading with realistic transaction costs
- Tracks equity, drawdowns, and position values
- Calculates comprehensive performance metrics
- Handles trade execution and tracking
- Provides detailed performance analysis

### Visualization Module

The `visualization.py` module:

- Creates Ichimoku cloud charts
- Visualizes trading signals and positions
- Generates performance dashboards
- Provides detailed component analysis
- Creates comparison charts for different strategies

## Performance Metrics

The system calculates the following performance metrics:

- Total and annualized returns
- Sharpe and Sortino ratios
- Maximum drawdown
- Win rate
- Profit factor
- Average trade P&L
- Monthly performance heatmap

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project uses the Gym framework for reinforcement learning environments
- Ichimoku Cloud strategy components are based on traditional technical analysis
- Thanks to all contributors who have helped improve this system