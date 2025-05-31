import pandas as pd
import matplotlib.pyplot as plt
import talib
import numpy as np
import pynance as pn
import os


class StockAnalyzer:
    def __init__(self, name, filepath):
        self.filepath = filepath
        self.name = name
        self.df = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.filepath)
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df.set_index('Date', inplace=True)

            # Ensure numeric types
            for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            print("Data loaded successfully.")
        except Exception as e:
            print(f"[Error] Failed to load data: {e}")

    def calculate_indicators(self):
        try:
            self.df['SMA_20'] = talib.SMA(self.df['Close'], timeperiod=20)
            self.df['RSI'] = talib.RSI(self.df['Close'], timeperiod=14)
            macd, macdsignal, _ = talib.MACD(
                self.df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
            self.df['MACD'] = macd
            self.df['MACD_Signal'] = macdsignal
            self.df['Daily_Return'] = self.df['Close'].pct_change()
            print("Technical indicators calculated.")
        except Exception as e:
            print(f"[Error] Failed to calculate indicators: {e}")

    def plot_sma(self):
        try:
            plt.figure(figsize=(14, 6))
            plt.plot(self.df['Close'], label='Close Price')
            plt.plot(self.df['SMA_20'], label='SMA 20', linestyle='--')
            plt.title(f"{self.name} Stock Price with 20-Day SMA")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True)
            plt.show()
        except Exception as e:
            print(f"[Error] Failed to plot SMA: {e}")

    def plot_rsi(self):
        try:
            plt.figure(figsize=(14, 4))
            plt.plot(self.df['RSI'], label='RSI', color='purple')
            plt.axhline(70, color='red', linestyle='--')
            plt.axhline(30, color='green', linestyle='--')
            plt.title("Relative Strength Index (RSI)")
            plt.xlabel("Date")
            plt.ylabel("RSI")
            plt.grid(True)
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"[Error] Failed to plot RSI: {e}")

    def plot_macd(self):
        try:
            plt.figure(figsize=(14, 4))
            plt.plot(self.df['MACD'], label='MACD', color='blue')
            plt.plot(self.df['MACD_Signal'], label='Signal Line', color='orange')
            plt.title("MACD and Signal Line")
            plt.xlabel("Date")
            plt.grid(True)
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"[Error] Failed to plot MACD: {e}")

    def plot_daily_return(self):
        try:
            plt.figure(figsize=(14, 4))
            plt.plot(self.df['Daily_Return'], label='Daily Return')
            plt.title("Daily Return %")
            plt.xlabel("Date")
            plt.ylabel("Return")
            plt.grid(True)
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"[Error] Failed to plot daily return: {e}")


class PortfolioAnalyzer:
    def __init__(self, ticker_file_map):
        self.ticker_file_map = ticker_file_map
        self.data = {}

    def load_data(self):
        print("Fetching data from CSV files...")
        for ticker, filepath in self.ticker_file_map.items():
            try:
                df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
                self.data[ticker] = df['Close']
                print(f"✔ Data loaded for {ticker}")
            except Exception as e:
                print(f" Failed to load data for {ticker}: {e}")

        if self.data:
            self.df = pd.DataFrame(self.data)
        else:
            print(" No valid data loaded from CSV files.")

    def calculate_metrics(self):
        try:
            self.returns = self.df.pct_change().dropna()
            self.annual_returns = self.returns.mean() * 252
            self.volatility = self.returns.std() * np.sqrt(252)
            self.sharpe_ratio = (self.annual_returns / self.volatility).round(2)

            self.portfolio_value = (1 + self.returns).cumprod()
            self.portfolio_value['Total'] = self.portfolio_value.mean(axis=1)
            print("✔ Metrics calculated successfully.")
        except Exception as e:
            print(f" Error calculating metrics: {e}")

    def print_summary(self):
        if not hasattr(self, 'annual_returns'):
            print(" Metrics not calculated. Please run `calculate_metrics()` first.")
            return
        print("Annual Returns:\n", self.annual_returns.round(4))
        print("\nVolatility:\n", self.volatility.round(4))
        print("\nSharpe Ratio:\n", self.sharpe_ratio)

    def plot_performance(self):
        if not hasattr(self, 'portfolio_value'):
            print(" Portfolio value not calculated. Please run `calculate_metrics()` first.")
            return
        try:
            plt.figure(figsize=(14, 6))
            for ticker in self.ticker_file_map:
                if ticker in self.portfolio_value:
                    plt.plot(self.portfolio_value[ticker], label=ticker)
            plt.plot(self.portfolio_value['Total'], label='Portfolio Avg', linewidth=2, color='black')
            plt.title("Portfolio vs Individual Stocks")
            plt.xlabel("Date")
            plt.ylabel("Value (Normalized)")
            plt.legend()
            plt.grid(True)
            plt.show()
        except Exception as e:
            print(f" Error plotting performance: {e}")



