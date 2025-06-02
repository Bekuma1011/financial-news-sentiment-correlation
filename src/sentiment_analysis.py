import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
import pytz



class SentimentReturnAnalyzer:
    """
    A class to analyze the correlation between financial news sentiment and stock returns.
    """

    def __init__(self, news_path, stock_path, stock_symbol):
        self.news_path = news_path
        self.stock_path = stock_path
        self.stock_symbol = stock_symbol
        self.news_df = None
        self.stock_df = None
        self.merged_df = None

    def __repr__(self):
        return f"<SentimentReturnAnalyzer stock={self.stock_symbol} records={len(self.merged_df) if self.merged_df is not None else 0}>"

    def load_data(self):
        """Loads and preprocesses news and stock data."""
        try:
            self.news_df = pd.read_csv(self.news_path)
            self.stock_df = pd.read_csv(self.stock_path)

            # Normalize column names
            self.news_df.columns = self.news_df.columns.str.lower()
            self.stock_df.columns = self.stock_df.columns.str.lower()

            # Filter news for the specific stock
            self.news_df = self.news_df[self.news_df["stock"] == self.stock_symbol]

            # Clean and convert date column in news_df
            self.news_df["date"] = (
                self.news_df["date"]
                .astype(str)
                .str.strip()
                .str.replace(r"\s{2,}", " ", regex=True)
            )
            self.news_df["date"] = pd.to_datetime(self.news_df["date"], errors="coerce")

                        # Handle unparsed dates with custom parser
            invalid_mask = self.news_df["date"].isna()
            if invalid_mask.any():
                reparsed_values = []
                for i in self.news_df.loc[invalid_mask].index:
                    parsed = self.try_parse_custom_date(self.news_df.at[i, "date"])
                    reparsed_values.append(parsed)

                # Convert the reparsed dates to a proper Series with the right dtype
                reparsed_series = pd.Series(reparsed_values, index=self.news_df.loc[invalid_mask].index)

                # Ensure 'date' column is naive (no timezone) before assigning
                self.news_df["date"] = self.news_df["date"].dt.tz_localize(None)

                # Assign the cleaned values
                self.news_df.loc[invalid_mask, "date"] = reparsed_series

            # Clean and parse stock_df date column
            self.stock_df["date"] = pd.to_datetime(self.stock_df["date"], errors="coerce")

            # Debug invalid dates
            invalid_dates = self.news_df[self.news_df["date"].isna()]
            print(f"[Debug] Unparsed news dates after fallback parsing: {len(invalid_dates)}")

        except Exception as e:
            print(f"[Error] Failed to load and process data: {e}")
            raise
    
    @staticmethod
    def try_parse_custom_date(date_str):
        """Attempts to parse non-standard date strings."""
        try:
            dt = datetime.strptime(date_str, "%m/%d/%Y %I:%M:%S %p")
            return dt  # naive datetime
        except Exception:
            return pd.NaT
        
    def analyze_sentiment(self):
        """Performs sentiment analysis on news headlines."""
        try:
            def get_sentiment(text):
                return TextBlob(text).sentiment.polarity

            self.news_df["sentiment"] = self.news_df["headline"].apply(get_sentiment)

        except Exception as e:
            print(f"[Error] Sentiment analysis failed: {e}")
            raise

    def aggregate_sentiment(self):
        """Averages sentiment scores per day."""
        try:
            return self.news_df.groupby("date")["sentiment"].mean().reset_index()
        except Exception as e:
            print(f"[Error] Failed to aggregate sentiment: {e}")
            raise

    def calculate_returns(self):
        """Calculates daily percentage stock returns."""
        try:
            self.stock_df.sort_values("date", inplace=True)
            self.stock_df["daily_return"] = self.stock_df["close"].pct_change()
        except Exception as e:
            print(f"[Error] Failed to compute daily returns: {e}")
            raise

    def merge_data(self):
        """Merges sentiment and return data on the date column."""
        try:
            daily_sentiment = self.aggregate_sentiment()

            self.stock_df["date"] = self.stock_df["date"].dt.normalize()
            daily_sentiment["date"] = pd.to_datetime(daily_sentiment["date"]).dt.normalize()

            self.merged_df = pd.merge(
                daily_sentiment,
                self.stock_df[["date", "daily_return"]],
                on="date",
                how="inner"
            )

            if self.merged_df.empty:
                print("[Warning] Merged DataFrame is empty. Check date alignment and input data.")
            else:
                print(f"[Info] Successfully merged data. Records: {len(self.merged_df)}")

        except Exception as e:
            print(f"[Error] Failed to merge sentiment and stock return data: {e}")
            raise

    def compute_correlation(self):
        """Computes Pearson correlation between sentiment and stock returns."""
        try:
            corr = self.merged_df["sentiment"].corr(self.merged_df["daily_return"])
            print(f"[Info] Pearson correlation: {corr:.4f}")
            return corr
        except Exception as e:
            print(f"[Error] Failed to compute correlation: {e}")
            raise

    def plot_relationship(self):
        """Visualizes the sentiment vs. daily return relationship."""
        try:
            plt.figure(figsize=(10, 6))
            sns.regplot(data=self.merged_df, x="sentiment", y="daily_return", scatter_kws={"alpha": 0.6})
            plt.title(f"Sentiment vs. Daily Return ({self.stock_symbol})")
            plt.xlabel("Average Daily Sentiment")
            plt.ylabel("Daily Stock Return")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"[Error] Failed to generate plot: {e}")
            raise
    def plot_correlation_heatmap(self):
        """Plots a heatmap of the correlation between sentiment and daily return."""
        try:
            correlation_matrix = self.merged_df[["sentiment", "daily_return"]].corr()
            plt.figure(figsize=(6, 4))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title("Correlation between Sentiment and Daily Stock Returns")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"[Error] Failed to plot correlation heatmap: {e}")
            raise    