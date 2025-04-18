import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class BacktestEngine:
    """
    A backtesting engine for evaluating trading strategies.
    """
    
    def __init__(self, initial_capital=10000, commission=0.001, slippage=0.0005):
        """
        Initialize the backtesting engine.
        
        Args:
            initial_capital (float): Initial capital for backtesting
            commission (float): Commission rate per trade (e.g., 0.001 = 0.1%)
            slippage (float): Slippage rate per trade (e.g., 0.0005 = 0.05%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.reset()
    
    def reset(self):
        """Reset the backtest state."""
        self.capital = self.initial_capital
        self.position = 0
        self.position_value = 0
        self.entry_price = 0
        self.trades = []
        self.daily_returns = []
        self.equity_curve = []
        self.max_equity = self.initial_capital
        self.max_drawdown = 0
        
    def calculate_transaction_costs(self, position_delta, price):
        """
        Calculate transaction costs including commission and slippage.
        
        Args:
            position_delta (float): Change in position size
            price (float): Current price
            
        Returns:
            float: Total transaction costs
        """
        # Commission is proportional to trade value
        commission_cost = abs(position_delta * price * self.commission)
        
        # Slippage is proportional to trade value
        slippage_cost = abs(position_delta * price * self.slippage)
        
        return commission_cost + slippage_cost
    
    def execute_trade(self, date, price, target_position, confidence=1.0):
        """
        Execute a trade to reach the target position.
        
        Args:
            date (datetime): Trade date
            price (float): Current price
            target_position (float): Target position after the trade
            confidence (float): Confidence level in the trade (0-1)
            
        Returns:
            dict: Trade information
        """
        position_delta = target_position - self.position
        
        if position_delta == 0:
            return None
        
        # Calculate transaction costs
        transaction_costs = self.calculate_transaction_costs(position_delta, price)
        
        # Calculate trade value
        trade_value = abs(position_delta * price)
        
        # Execute the trade
        self.position = target_position
        self.position_value = self.position * price
        
        # Adjust capital
        self.capital -= position_delta * price
        self.capital -= transaction_costs
        
        # Record the trade
        trade_info = {
            'date': date,
            'price': price,
            'position_delta': position_delta,
            'position': self.position,
            'transaction_costs': transaction_costs,
            'trade_value': trade_value,
            'confidence': confidence,
            'capital': self.capital,
            'equity': self.capital + self.position_value
        }
        
        # If it's an entry, record entry price
        if self.position != 0 and self.entry_price == 0:
            self.entry_price = price
            trade_info['trade_type'] = 'entry'
        # If it's an exit, calculate profit/loss
        elif self.position == 0 and self.entry_price != 0:
            trade_pnl = (price - self.entry_price) * position_delta
            trade_info['trade_type'] = 'exit'
            trade_info['entry_price'] = self.entry_price
            trade_info['pnl'] = trade_pnl
            trade_info['pnl_pct'] = trade_pnl / (self.entry_price * abs(position_delta))
            # Reset entry price
            self.entry_price = 0
        # If it's a position increase
        elif abs(self.position) > abs(self.position - position_delta):
            trade_info['trade_type'] = 'increase'
        # If it's a position decrease
        elif abs(self.position) < abs(self.position - position_delta):
            trade_info['trade_type'] = 'decrease'
        # If it's a position reversal
        elif np.sign(self.position) != np.sign(self.position - position_delta):
            trade_info['trade_type'] = 'reversal'
        
        self.trades.append(trade_info)
        return trade_info
    
    def update_daily_stats(self, date, row):
        """
        Update daily statistics.
        
        Args:
            date (datetime): Current date
            row (pd.Series): Current data row
        """
        # Update position value
        self.position_value = self.position * row['close']
        
        # Calculate equity
        equity = self.capital + self.position_value
        
        # Calculate daily return
        previous_equity = self.equity_curve[-1]['equity'] if self.equity_curve else self.initial_capital
        daily_return = (equity / previous_equity) - 1
        
        # Update max equity and drawdown
        self.max_equity = max(self.max_equity, equity)
        current_drawdown = (self.max_equity - equity) / self.max_equity
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Record daily stats
        daily_stats = {
            'date': date,
            'equity': equity,
            'cash': self.capital,
            'position': self.position,
            'position_value': self.position_value,
            'daily_return': daily_return,
            'drawdown': current_drawdown
        }
        
        self.equity_curve.append(daily_stats)
        self.daily_returns.append(daily_return)
    
    def run_backtest(self, df, positions, confidence=None):
        """
        Run a backtest on a dataframe with precalculated positions.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data
            positions (pd.Series): Series of target positions
            confidence (pd.Series): Optional series of confidence values (0-1)
            
        Returns:
            pd.DataFrame: DataFrame with backtest results
        """
        # Reset backtest state
        self.reset()
        
        # Default confidence to 1.0 if not provided
        if confidence is None:
            confidence = pd.Series(1.0, index=df.index)
        
        # Iterate through the data
        for i in range(len(df)):
            date = df.index[i]
            row = df.iloc[i]
            
            # Execute trade if position changes
            current_position = self.position
            target_position = positions.iloc[i]
            
            if target_position != current_position:
                self.execute_trade(
                    date=date,
                    price=row['close'],
                    target_position=target_position,
                    confidence=confidence.iloc[i]
                )
            
            # Update daily stats
            self.update_daily_stats(date, row)
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('date', inplace=True)
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        if not trades_df.empty:
            trades_df.set_index('date', inplace=True)
        
        # Merge with original data
        result = df.copy()
        result['position'] = positions
        
        if not equity_df.empty:
            for col in equity_df.columns:
                result[col] = equity_df[col]
        
        return result, trades_df
    
    def calculate_performance_metrics(self, equity_curve):
        """
        Calculate performance metrics based on equity curve.
        
        Args:
            equity_curve (pd.DataFrame): DataFrame with equity curve data
            
        Returns:
            dict: Dictionary of performance metrics
        """
        # Convert to DataFrame if it's a list
        if isinstance(equity_curve, list):
            equity_curve = pd.DataFrame(equity_curve)
            if 'date' in equity_curve.columns:
                equity_curve.set_index('date', inplace=True)
        
        # Extract daily returns
        returns = equity_curve['daily_return'].dropna()
        
        # Calculate key metrics
        total_days = len(returns)
        trading_days_per_year = 252
        
        # Total return
        total_return = (equity_curve['equity'].iloc[-1] / self.initial_capital) - 1
        
        # Annualized return
        annualized_return = (1 + total_return) ** (trading_days_per_year / total_days) - 1
        
        # Volatility
        daily_volatility = returns.std()
        annualized_volatility = daily_volatility * np.sqrt(trading_days_per_year)
        
        # Sharpe Ratio (assuming risk-free rate of 0)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Sortino Ratio (downside risk only)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(trading_days_per_year)
        sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0
        
        # Max Drawdown
        max_drawdown = equity_curve['drawdown'].max()
        
        # Calmar Ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Win rate
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        if not trades_df.empty and 'pnl' in trades_df.columns:
            winning_trades = (trades_df['pnl'] > 0).sum()
            total_trades = len(trades_df[trades_df['trade_type'] == 'exit'])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Profit factor
            gross_profits = trades_df.loc[trades_df['pnl'] > 0, 'pnl'].sum() if winning_trades > 0 else 0
            gross_losses = abs(trades_df.loc[trades_df['pnl'] < 0, 'pnl'].sum()) if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
            profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
            
            # Average trade
            avg_trade = trades_df['pnl'].mean() if 'pnl' in trades_df.columns else 0
            avg_win = trades_df.loc[trades_df['pnl'] > 0, 'pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df.loc[trades_df['pnl'] < 0, 'pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
        else:
            win_rate = 0
            profit_factor = 0
            avg_trade = 0
            avg_win = 0
            avg_loss = 0
            total_trades = 0
        
        # Compile metrics
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'avg_trade': avg_trade,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
        
        return metrics
    
    def run_strategy_backtest(self, df, strategy_function, strategy_params=None):
        """
        Run a backtest using a strategy function.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data
            strategy_function (callable): Function that returns positions
            strategy_params (dict): Parameters for the strategy function
            
        Returns:
            tuple: (results DataFrame, trades DataFrame, performance metrics)
        """
        # Apply strategy function to get positions
        if strategy_params is None:
            positions, confidence = strategy_function(df)
        else:
            positions, confidence = strategy_function(df, **strategy_params)
        
        # Run backtest
        results, trades = self.run_backtest(df, positions, confidence)
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(results)
        
        return results, trades, metrics
    
    def plot_equity_curve(self, results, benchmark=None):
        """
        Plot equity curve and drawdowns.
        
        Args:
            results (pd.DataFrame): DataFrame with backtest results
            benchmark (pd.Series): Optional benchmark returns
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        plt.figure(figsize=(15, 10))
        
        # Plot equity curve
        plt.subplot(2, 1, 1)
        plt.plot(results.index, results['equity'], label='Strategy')
        
        if benchmark is not None:
            # Normalize benchmark to same starting point
            benchmark_equity = self.initial_capital * (1 + benchmark.cumsum())
            plt.plot(results.index, benchmark_equity, label='Benchmark', alpha=0.7)
        
        plt.title('Equity Curve')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        plt.legend()
        
        # Plot drawdowns
        plt.subplot(2, 1, 2)
        plt.fill_between(results.index, 0, results['drawdown'], color='red', alpha=0.3)
        plt.title('Drawdowns')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_trade_analysis(self, trades):
        """
        Plot trade analysis charts.
        
        Args:
            trades (pd.DataFrame): DataFrame with trade information
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if trades.empty or 'pnl' not in trades.columns:
            return None
        
        plt.figure(figsize=(15, 15))
        
        # Plot trade P&L
        plt.subplot(3, 1, 1)
        trades['pnl'].plot(kind='bar', color=trades['pnl'].apply(lambda x: 'green' if x > 0 else 'red'))
        plt.title('Trade P&L')
        plt.ylabel('P&L ($)')
        plt.grid(True)
        
        # Plot trade P&L distribution
        plt.subplot(3, 1, 2)
        sns.histplot(trades['pnl'], kde=True)
        plt.title('P&L Distribution')
        plt.xlabel('P&L ($)')
        plt.grid(True)
        
        # Plot cumulative P&L
        plt.subplot(3, 1, 3)
        trades['cumulative_pnl'] = trades['pnl'].cumsum()
        trades['cumulative_pnl'].plot()
        plt.title('Cumulative P&L')
        plt.ylabel('Cumulative P&L ($)')
        plt.grid(True)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_monthly_returns(self, results):
        """
        Plot monthly returns heatmap.
        
        Args:
            results (pd.DataFrame): DataFrame with backtest results
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Resample daily returns to monthly
        monthly_returns = results['daily_return'].resample('M').apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # Create a pivot table with years as rows and months as columns
        monthly_returns.index = monthly_returns.index.to_period('M')
        monthly_return_table = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first()
        monthly_return_table = monthly_return_table.unstack()
        
        # Plot heatmap
        plt.figure(figsize=(15, 8))
        sns.heatmap(monthly_return_table, annot=True, cmap='RdYlGn', fmt='.1%',
                   linewidths=1, center=0, vmin=-0.1, vmax=0.1)
        plt.title('Monthly Returns Heatmap')
        plt.ylabel('Year')
        plt.xlabel('Month')
        
        return plt.gcf()