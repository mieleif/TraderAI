# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands
- Setup environment: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
- Run PPO training: `cd PPO && python main.py --mode train --data ../data/ETHUSDT_4h_data.csv --sample_size 500`
- Run PPO backtest: `cd PPO && python main.py --mode backtest --data ../data/ETHUSDT_4h_data.csv --visualize`
- Run PPO hybrid mode: `cd PPO && python main.py --mode hybrid --data ../data/ETHUSDT_4h_data.csv --visualize`
- Run basic tests: `cd PPO && python test_script.py` 
- Note: The DQN implementation is incomplete and missing modules like `environments.enhanced_trading_env`

## Project Structure
- The `data/` directory contains all datasets and data utilities
- Virtual environment and requirements are centralized in the project root
- The `PPO/` directory contains the main implementation
- The `DQN/` directory contains an alternative implementation (incomplete)

## Data Files
- `data/ETHUSDT_4h_data.csv` - Primary dataset with 4-hour ETHUSDT data and Ichimoku indicators
- `data/ETHUSDT_4_hours_data.csv` - Alternative format used by enhanced_training_pipeline.py
- `data/alternative_approach.py` - Utility for fetching data from Binance API

## Code Style Guidelines
- Imports: group standard library imports first, then third party, then local
- Docstrings: use triple-quoted multi-line docstrings with parameter descriptions
- Logging: use the provided logger setup in files instead of print statements
- Error handling: use try/except with specific exception types
- Types: use numeric datatypes (float, int) in ML models, handle NaN values properly
- Naming: use snake_case for variables and functions, CamelCase for classes
- Data processing: normalize features before using in neural networks
- File formatting: ensure proper indentation (4 spaces) for Python files
- Sample sizes: Use at least 500 samples when running with reduced dataset sizes to ensure adequate data for Ichimoku calculations