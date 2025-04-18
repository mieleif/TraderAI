import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.patches import Rectangle
import mplfinance as mpf
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta

class TradingVisualization:
    """
    A class for visualizing trading data, Ichimoku clouds, and trading signals.
    """
    
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        """
        Initialize the visualization class.
        
        Args:
            style (str): Matplotlib style to use
        """
        plt.style.use(style)
        self.colors = {
            'price': '#1f77b4',
            'tenkan': '#ff7f0e',
            'kijun': '#2ca02c',
            'senkou_a': '#d62728',
            'senkou_b': '#9467bd',
            'chikou': '#8c564b',
            'buy': '#2ca02c',
            'sell': '#d62728',
            'cloud_green': (0.565, 0.933, 0.565, 0.2),  # Light green with alpha
            'cloud_red': (1.0, 0.412, 0.38, 0.2)        # Light red with alpha
        }
    
    def plot_ichimoku_cloud(self, df, figsize=(15, 10), show_volume=True, show_signals=True, output_file=None):
        """
        Plot price data with Ichimoku cloud overlay.
        
        Args:
            df (pd.DataFrame): DataFrame with price data and Ichimoku components
            figsize (tuple): Figure size (width, height)
            show_volume (bool): Whether to show volume subplot
            show_signals (bool): Whether to show trading signals
            output_file (str): Optional path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Check if df is datetime indexed
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        
        # Check if all required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        ichimoku_cols = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']
        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"DataFrame is missing required column: {col}")
        
        # Create figure and subplots
        fig = plt.figure(figsize=figsize)
        
        if show_volume and 'volume' in df.columns:
            gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1], sharex=ax1)
        else:
            ax1 = plt.subplot(1, 1, 1)
            ax2 = None
        
        # Plot candlestick chart
        mpf.plot(df, type='candle', style='charles', ax=ax1, volume=False, show_nontrading=False)
        
        # Set x-axis labels
        ax1.set_xlabel('')
        date_format = mdates.DateFormatter('%Y-%m-%d')
        ax1.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
        
        # Plot Ichimoku components if available
        if all(col in df.columns for col in ichimoku_cols):
            # Plot Tenkan-sen (Conversion Line)
            ax1.plot(df.index, df['tenkan_sen'], color=self.colors['tenkan'], 
                     linewidth=1.5, label='Tenkan-sen')
            
            # Plot Kijun-sen (Base Line)
            ax1.plot(df.index, df['kijun_sen'], color=self.colors['kijun'], 
                     linewidth=1.5, label='Kijun-sen')
            
            # Plot Chikou Span (Lagging Span)
            chikou_data = df['chikou_span'].dropna()
            ax1.plot(chikou_data.index, chikou_data, color=self.colors['chikou'], 
                     linewidth=1, alpha=0.7, label='Chikou Span')
            
            # Plot Senkou Span A and B (Leading Spans)
            span_a = df['senkou_span_a'].dropna()
            span_b = df['senkou_span_b'].dropna()
            
            ax1.plot(span_a.index, span_a, color=self.colors['senkou_a'], 
                     linewidth=1, alpha=0.8, label='Senkou Span A')
            ax1.plot(span_b.index, span_b, color=self.colors['senkou_b'], 
                     linewidth=1, alpha=0.8, label='Senkou Span B')
            
            # Fill cloud (green when Senkou A > Senkou B, red otherwise)
            # Make sure indices match by reindexing to the same index
            common_index = span_a.index.intersection(span_b.index)
            span_a_aligned = span_a.loc[common_index]
            span_b_aligned = span_b.loc[common_index]
            
            cloud_green = span_a_aligned > span_b_aligned
            cloud_red = span_a_aligned <= span_b_aligned
            
            ax1.fill_between(common_index, span_a_aligned, span_b_aligned, 
                             where=cloud_green, color=self.colors['cloud_green'])
            ax1.fill_between(common_index, span_a_aligned, span_b_aligned, 
                             where=cloud_red, color=self.colors['cloud_red'])
        
        # Plot trading signals if available and requested
        if show_signals:
            if 'position' in df.columns:
                # Find buy signals (position changes from 0 or negative to positive)
                buy_signals = df[(df['position'] > 0) & 
                                 (df['position'].shift(1).fillna(0) <= 0)]
                
                # Find sell signals (position changes from 0 or positive to negative)
                sell_signals = df[(df['position'] < 0) & 
                                  (df['position'].shift(1).fillna(0) >= 0)]
                
                # Plot buy and sell signals
                ax1.scatter(buy_signals.index, buy_signals['low'] * 0.99, 
                            marker='^', color=self.colors['buy'], s=100, label='Buy')
                ax1.scatter(sell_signals.index, sell_signals['high'] * 1.01, 
                            marker='v', color=self.colors['sell'], s=100, label='Sell')
            
            elif 'signal' in df.columns:
                # Assuming signal column with values 1 (buy), -1 (sell), 0 (hold)
                buy_signals = df[df['signal'] == 1]
                sell_signals = df[df['signal'] == -1]
                
                # Plot buy and sell signals
                ax1.scatter(buy_signals.index, buy_signals['low'] * 0.99, 
                            marker='^', color=self.colors['buy'], s=100, label='Buy')
                ax1.scatter(sell_signals.index, sell_signals['high'] * 1.01, 
                            marker='v', color=self.colors['sell'], s=100, label='Sell')
        
        # Add legend
        ax1.legend(loc='upper left')
        
        # Plot volume if requested and available
        if show_volume and ax2 is not None and 'volume' in df.columns:
            # Plot volume directly instead of using mpf
            ax2.bar(df.index, df['volume'], color='#1f77b4', alpha=0.5)
            ax2.set_ylabel('Volume')
        
        # Set titles and labels
        instrument = df.get('instrument', 'Instrument')
        if isinstance(instrument, pd.Series):
            instrument = instrument.iloc[0] if not instrument.empty else 'Instrument'
        
        timeframe = df.get('timeframe', '')
        if isinstance(timeframe, pd.Series):
            timeframe = timeframe.iloc[0] if not timeframe.empty else ''
        
        title = f'Ichimoku Cloud Chart - {instrument} {timeframe}'
        ax1.set_title(title)
        ax1.set_ylabel('Price')
        
        plt.tight_layout()
        
        # Save if output_file is provided
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_signal_comparison(self, df, figsize=(15, 12), output_file=None):
        """
        Plot a comparison of AI signals, Ichimoku signals, and combined signals.
        
        Args:
            df (pd.DataFrame): DataFrame with signals and price data
            figsize (tuple): Figure size (width, height)
            output_file (str): Optional path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Check if required columns exist
        required_cols = ['close', 'ai_signal', 'ichimoku_signal', 'combined_signal']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"DataFrame is missing required columns: {', '.join(missing_cols)}")
        
        # Create figure and subplots
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])
        
        # Price chart
        ax1 = plt.subplot(gs[0])
        ax1.plot(df.index, df['close'], color=self.colors['price'], label='Price')
        ax1.set_title('Price Chart with Signals')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        
        # Plot buy/sell positions if available
        if 'position' in df.columns:
            # Position line at bottom of price chart
            position_line = ax1.twinx()
            position_line.fill_between(df.index, 0, df['position'], 
                                      where=df['position'] > 0, color=self.colors['buy'], alpha=0.3)
            position_line.fill_between(df.index, 0, df['position'], 
                                      where=df['position'] < 0, color=self.colors['sell'], alpha=0.3)
            position_line.set_ylabel('Position')
            position_line.set_ylim(-1.5, 1.5)  # Adjust based on your position sizing
        
        # Signal comparison
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.plot(df.index, df['ai_signal'], color='blue', label='AI Signal')
        ax2.plot(df.index, df['ichimoku_signal'], color='green', label='Ichimoku Signal')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax2.set_ylabel('Signal [-1,1]')
        ax2.set_ylim(-1.2, 1.2)  # Fixed y-axis for signals
        ax2.legend(loc='upper left')
        
        # Combined signal with confidence
        ax3 = plt.subplot(gs[2], sharex=ax1)
        
        # Plot combined signal
        ax3.plot(df.index, df['combined_signal'], color='purple', label='Combined Signal')
        ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Plot confidence if available
        if 'confidence' in df.columns:
            confidence = ax3.twinx()
            confidence.fill_between(df.index, 0, df['confidence'], color='gray', alpha=0.2)
            confidence.set_ylabel('Confidence')
            confidence.set_ylim(0, 1.1)  # Confidence is between 0 and 1
        
        ax3.set_ylabel('Signal [-1,1]')
        ax3.set_ylim(-1.2, 1.2)  # Fixed y-axis for signals
        ax3.legend(loc='upper left')
        
        # Set common properties
        ax3.set_xlabel('Date')
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        # Save if output_file is provided
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_performance_dashboard(self, results, trades=None, metrics=None, benchmark=None, figsize=(15, 15), output_file=None):
        """
        Create a comprehensive performance dashboard.
        
        Args:
            results (pd.DataFrame): DataFrame with backtest results
            trades (pd.DataFrame): DataFrame with trade information
            metrics (dict): Dictionary of performance metrics
            benchmark (pd.Series): Optional benchmark returns
            figsize (tuple): Figure size (width, height)
            output_file (str): Optional path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if 'equity' not in results.columns:
            raise ValueError("Results DataFrame must contain an 'equity' column")
        
        # Create figure and subplots
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(4, 2, height_ratios=[2, 1, 1, 1])
        
        # 1. Equity curve
        ax_equity = plt.subplot(gs[0, :])
        ax_equity.plot(results.index, results['equity'], color='blue', label='Strategy')
        
        if benchmark is not None:
            # Normalize benchmark to same starting point
            initial_capital = results['equity'].iloc[0]
            benchmark_equity = initial_capital * (1 + benchmark.cumsum())
            ax_equity.plot(results.index, benchmark_equity, color='gray', 
                           linestyle='--', label='Benchmark', alpha=0.7)
        
        ax_equity.set_title('Equity Curve')
        ax_equity.set_ylabel('Equity ($)')
        ax_equity.legend(loc='upper left')
        ax_equity.grid(True)
        
        # 2. Drawdowns
        ax_dd = plt.subplot(gs[1, :], sharex=ax_equity)
        if 'drawdown' in results.columns:
            ax_dd.fill_between(results.index, 0, results['drawdown'] * 100, 
                              color='red', alpha=0.3)
            ax_dd.set_ylabel('Drawdown (%)')
            ax_dd.set_title('Drawdowns')
            ax_dd.grid(True)
        
        # 3. Position size over time
        ax_pos = plt.subplot(gs[2, 0], sharex=ax_equity)
        if 'position' in results.columns:
            ax_pos.fill_between(results.index, 0, results['position'], 
                               where=results['position'] > 0, color='green', alpha=0.3)
            ax_pos.fill_between(results.index, 0, results['position'], 
                               where=results['position'] < 0, color='red', alpha=0.3)
            ax_pos.set_ylabel('Position')
            ax_pos.set_title('Position Size')
            ax_pos.grid(True)
        
        # 4. Monthly returns heatmap (or simple text if datetime index not available)
        ax_monthly = plt.subplot(gs[2, 1])
        
        if 'daily_return' in results.columns:
            # Check if we have a datetime index to create monthly returns
            if isinstance(results.index, pd.DatetimeIndex):
                try:
                    # Resample daily returns to monthly
                    monthly_returns = results['daily_return'].resample('ME').apply(
                        lambda x: (1 + x).prod() - 1
                    )
                    
                    # Create a pivot table with years as rows and months as columns
                    monthly_returns.index = monthly_returns.index.to_period('M')
                    pivot_data = []
                    
                    for date, value in monthly_returns.items():
                        pivot_data.append({
                            'Year': date.year,
                            'Month': date.month,
                            'Return': value
                        })
                    
                    pivot_df = pd.DataFrame(pivot_data)
                    if not pivot_df.empty:
                        pivot_table = pivot_df.pivot(index='Year', columns='Month', values='Return')
                        
                        # Plot heatmap
                        sns.heatmap(pivot_table, ax=ax_monthly, cmap='RdYlGn', center=0, 
                                    annot=True, fmt='.1%', cbar=False)
                        ax_monthly.set_title('Monthly Returns')
                    else:
                        ax_monthly.text(0.5, 0.5, 'Insufficient data for monthly returns',
                                      horizontalalignment='center', verticalalignment='center')
                except Exception as e:
                    ax_monthly.text(0.5, 0.5, f'Could not create monthly returns: {str(e)}',
                                  horizontalalignment='center', verticalalignment='center')
            else:
                # If not datetime index, just show average monthly return
                avg_return = results['daily_return'].mean() * 30  # Approximate monthly
                ax_monthly.text(0.5, 0.5, f'Avg. Monthly Return: {avg_return:.2%}',
                               horizontalalignment='center', verticalalignment='center')
                ax_monthly.set_title('Monthly Returns (Approximated)')
        else:
            ax_monthly.text(0.5, 0.5, 'Daily returns data not available',
                           horizontalalignment='center', verticalalignment='center')
        
        # 5. Trade P&L (if trades are provided)
        ax_pnl = plt.subplot(gs[3, 0])
        if trades is not None and not trades.empty and 'pnl' in trades.columns:
            trade_nums = range(1, len(trades) + 1)
            colors = ['green' if pnl > 0 else 'red' for pnl in trades['pnl']]
            ax_pnl.bar(trade_nums, trades['pnl'], color=colors)
            ax_pnl.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax_pnl.set_title('Trade P&L')
            ax_pnl.set_xlabel('Trade Number')
            ax_pnl.set_ylabel('P&L ($)')
            ax_pnl.grid(True)
        else:
            ax_pnl.text(0.5, 0.5, 'No trade data available',
                       horizontalalignment='center', verticalalignment='center')
        
        # 6. Performance metrics table
        ax_metrics = plt.subplot(gs[3, 1])
        ax_metrics.axis('off')
        
        if metrics is not None:
            metrics_to_display = [
                ('Total Return', f"{metrics.get('total_return', 0)*100:.2f}%"),
                ('Annualized Return', f"{metrics.get('annualized_return', 0)*100:.2f}%"),
                ('Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"),
                ('Sortino Ratio', f"{metrics.get('sortino_ratio', 0):.2f}"),
                ('Max Drawdown', f"{metrics.get('max_drawdown', 0)*100:.2f}%"),
                ('Win Rate', f"{metrics.get('win_rate', 0)*100:.2f}%"),
                ('Profit Factor', f"{metrics.get('profit_factor', 0):.2f}"),
                ('Total Trades', f"{metrics.get('total_trades', 0)}")
            ]
            
            # Create table data
            cell_text = [[m[1]] for m in metrics_to_display]
            row_labels = [m[0] for m in metrics_to_display]
            
            # Add table
            table = ax_metrics.table(cellText=cell_text, rowLabels=row_labels, 
                                    colWidths=[0.6], loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            ax_metrics.set_title('Performance Metrics')
        else:
            ax_metrics.text(0.5, 0.5, 'No metrics data available',
                           horizontalalignment='center', verticalalignment='center')
        
        # Set common properties
        plt.tight_layout()
        fig.autofmt_xdate()
        
        # Save if output_file is provided
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_ichimoku_components_plot(self, df, figsize=(15, 20), output_file=None):
        """
        Create a detailed plot showing all Ichimoku components separately.
        
        Args:
            df (pd.DataFrame): DataFrame with price and Ichimoku data
            figsize (tuple): Figure size (width, height)
            output_file (str): Optional path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Check if required columns exist
        required_cols = ['close', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"DataFrame is missing required Ichimoku columns: {', '.join(missing_cols)}")
        
        # Create figure and subplots
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(6, 1, height_ratios=[3, 1, 1, 1, 1, 1])
        
        # 1. Price with Cloud
        ax1 = plt.subplot(gs[0])
        ax1.plot(df.index, df['close'], color=self.colors['price'], label='Price')
        
        span_a = df['senkou_span_a'].dropna()
        span_b = df['senkou_span_b'].dropna()
        
        ax1.plot(span_a.index, span_a, color=self.colors['senkou_a'], 
                 linewidth=1, alpha=0.8, label='Senkou Span A')
        ax1.plot(span_b.index, span_b, color=self.colors['senkou_b'], 
                 linewidth=1, alpha=0.8, label='Senkou Span B')
        
        # Fill cloud
        # Make sure indices match by reindexing to the same index
        common_index = span_a.index.intersection(span_b.index)
        span_a_aligned = span_a.loc[common_index]
        span_b_aligned = span_b.loc[common_index]
        
        cloud_green = span_a_aligned > span_b_aligned
        cloud_red = span_a_aligned <= span_b_aligned
        
        ax1.fill_between(common_index, span_a_aligned, span_b_aligned, 
                         where=cloud_green, color=self.colors['cloud_green'])
        ax1.fill_between(common_index, span_a_aligned, span_b_aligned, 
                         where=cloud_red, color=self.colors['cloud_red'])
        
        ax1.set_title('Price with Kumo (Cloud)')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # 2. Tenkan-sen
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.plot(df.index, df['close'], color=self.colors['price'], alpha=0.5, label='Price')
        ax2.plot(df.index, df['tenkan_sen'], color=self.colors['tenkan'], label='Tenkan-sen')
        ax2.set_title('Tenkan-sen (Conversion Line)')
        ax2.set_ylabel('Price')
        ax2.legend(loc='upper left')
        ax2.grid(True)
        
        # 3. Kijun-sen
        ax3 = plt.subplot(gs[2], sharex=ax1)
        ax3.plot(df.index, df['close'], color=self.colors['price'], alpha=0.5, label='Price')
        ax3.plot(df.index, df['kijun_sen'], color=self.colors['kijun'], label='Kijun-sen')
        ax3.set_title('Kijun-sen (Base Line)')
        ax3.set_ylabel('Price')
        ax3.legend(loc='upper left')
        ax3.grid(True)
        
        # 4. TK Cross
        ax4 = plt.subplot(gs[3], sharex=ax1)
        ax4.plot(df.index, df['tenkan_sen'], color=self.colors['tenkan'], label='Tenkan-sen')
        ax4.plot(df.index, df['kijun_sen'], color=self.colors['kijun'], label='Kijun-sen')
        
        # Highlight TK crosses
        tk_cross_bull = ((df['tenkan_sen'] > df['kijun_sen']) & 
                         (df['tenkan_sen'].shift(1) <= df['kijun_sen'].shift(1)))
        tk_cross_bear = ((df['tenkan_sen'] < df['kijun_sen']) & 
                         (df['tenkan_sen'].shift(1) >= df['kijun_sen'].shift(1)))
        
        for idx in df[tk_cross_bull].index:
            ax4.axvline(x=idx, color='green', alpha=0.5, linestyle='--')
        
        for idx in df[tk_cross_bear].index:
            ax4.axvline(x=idx, color='red', alpha=0.5, linestyle='--')
        
        ax4.set_title('TK Cross (Tenkan-Kijun Crossover)')
        ax4.set_ylabel('Price')
        ax4.legend(loc='upper left')
        ax4.grid(True)
        
        # 5. Price-Kumo Relationship
        ax5 = plt.subplot(gs[4], sharex=ax1)
        ax5.plot(df.index, df['close'], color=self.colors['price'], label='Price')
        ax5.plot(span_a.index, span_a, color=self.colors['senkou_a'], 
                 linewidth=1, alpha=0.8, label='Senkou Span A')
        ax5.plot(span_b.index, span_b, color=self.colors['senkou_b'], 
                 linewidth=1, alpha=0.8, label='Senkou Span B')
        
        # Fill cloud with lower opacity
        # Make sure indices match by reindexing to the same index
        common_index = span_a.index.intersection(span_b.index)
        span_a_aligned = span_a.loc[common_index]
        span_b_aligned = span_b.loc[common_index]
        
        cloud_green_aligned = span_a_aligned > span_b_aligned
        cloud_red_aligned = span_a_aligned <= span_b_aligned
        
        ax5.fill_between(common_index, span_a_aligned, span_b_aligned, 
                         where=cloud_green_aligned, color=self.colors['cloud_green'], alpha=0.2)
        ax5.fill_between(common_index, span_a_aligned, span_b_aligned, 
                         where=cloud_red_aligned, color=self.colors['cloud_red'], alpha=0.2)
        
        # Highlight when price breaks through the cloud
        kumo_upper = df[['senkou_span_a', 'senkou_span_b']].max(axis=1)
        kumo_lower = df[['senkou_span_a', 'senkou_span_b']].min(axis=1)
        
        break_up = ((df['close'] > kumo_upper) & 
                   (df['close'].shift(1) <= kumo_upper.shift(1)))
        break_down = ((df['close'] < kumo_lower) & 
                      (df['close'].shift(1) >= kumo_lower.shift(1)))
        
        for idx in df[break_up].index:
            ax5.axvline(x=idx, color='green', alpha=0.5, linestyle='--')
        
        for idx in df[break_down].index:
            ax5.axvline(x=idx, color='red', alpha=0.5, linestyle='--')
        
        ax5.set_title('Price-Kumo Relationship (Cloud Breakouts)')
        ax5.set_ylabel('Price')
        ax5.legend(loc='upper left')
        ax5.grid(True)
        
        # 6. Chikou Span
        ax6 = plt.subplot(gs[5], sharex=ax1)
        ax6.plot(df.index, df['close'], color=self.colors['price'], alpha=0.5, label='Price')
        ax6.plot(df.index, df['chikou_span'], color=self.colors['chikou'], label='Chikou Span')
        
        # Highlight when Chikou span crosses price
        chikou_cross_up = ((df['chikou_span'] > df['close']) & 
                          (df['chikou_span'].shift(1) <= df['close'].shift(1)))
        chikou_cross_down = ((df['chikou_span'] < df['close']) & 
                            (df['chikou_span'].shift(1) >= df['close'].shift(1)))
        
        for idx in df[chikou_cross_up].index:
            ax6.axvline(x=idx, color='green', alpha=0.5, linestyle='--')
        
        for idx in df[chikou_cross_down].index:
            ax6.axvline(x=idx, color='red', alpha=0.5, linestyle='--')
        
        ax6.set_title('Chikou Span (Lagging Span)')
        ax6.set_ylabel('Price')
        ax6.set_xlabel('Date')
        ax6.legend(loc='upper left')
        ax6.grid(True)
        
        # Set common properties
        plt.tight_layout()
        fig.autofmt_xdate()
        
        # Save if output_file is provided
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        return fig