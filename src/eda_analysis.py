import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords

class EdaAnalysis:
    def __init__(self, input_path, selected_stocks=None):
        self.input_path = input_path
        self.selected_stocks = selected_stocks
        self.df = pd.DataFrame()

    def load_data(self):
        try:
            self.df = pd.read_csv(self.input_path)
            print("File loaded successfully. Shape:", self.df.shape)

            self.df = self.df[self.df['stock'].isin(self.selected_stocks)]
            print(f"Filtered DataFrame shape of {self.selected_stocks}:", self.df.shape)
        except FileNotFoundError:
            print("File not found. Check the relative path.")
        except pd.errors.ParserError:
            print("File found but not a valid CSV.")
        except Exception as e:
            print("An unexpected error occurred:", str(e))

    def dataset_overview(self):
        print("\n =================== Dataset Overview:===================================")
        print(self.df.head(10))
        print("\n ============================== Last 5 Rows of the Dataset:========================")
        print(self.df.tail())
        print("\n ================================ Random Sample of 5 Rows:=============================")
        print(self.df.sample(5))
        print("\n =================================== Dataset Shape:=====================================")
        print(self.df.shape)
        print("\n The description of the numeric columns:")
        print(self.df.describe())
        print("Check for missing values:")
        print(self.df.isna().sum())

    def analyze_headline_lengths(self):
        self.df['headline_length'] = self.df['headline'].apply(len)
        print("\nHeadline length statistics:")
        print(self.df['headline_length'].describe())
    
    def check_missing_value(self):
        print("check for missing values")
        print(self.df.isna().sum())
    
    def count_number_article(self):
        publisher_counts = self.df['publisher'].value_counts()
        print(publisher_counts)

    def top_publishers_plot(self):
        publisher_counts = self.df['publisher'].value_counts()
        print("\nTop 10 publishers:")
        print(publisher_counts.head(10))
        sns.barplot(x=publisher_counts.values[:10], y=publisher_counts.index[:10])
        plt.title("Top 10 Publishers by Article Count")
        plt.xlabel("Article Count")
        plt.ylabel("Publisher")
        plt.tight_layout()
        plt.show()

    def parse_dates(self):
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce', utc=True)
        self.df['date_only'] = self.df['date'].dt.date

    def daily_article_trend(self):
        # Ensure 'date' column is in datetime format
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce', utc=True)

        # Extract only date part (YYYY-MM-DD)
        self.df['date_only'] = self.df['date'].dt.date

        # Count articles published per day
        daily_counts = self.df.groupby('date_only').size()

        # Plot daily article count
        plt.figure(figsize=(14, 6))
        daily_counts.plot(title=" Daily Article Count Over Time", grid=True)
        plt.xlabel("Date")
        plt.ylabel("Number of Articles")
        plt.tight_layout()
        plt.show()

    def article_hour_distribution(self):
        self.df['hour'] = self.df['date'].dt.hour
        hourly_counts = self.df['hour'].value_counts().sort_index()
        plt.figure(figsize=(10, 5))
        hourly_counts.plot(kind='bar', title="⏰ Article Publication by Hour of Day", color='skyblue')
        plt.xlabel("Hour of Day (0-23)")
        plt.ylabel("Number of Articles")
        plt.grid(True)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def text_analysis(self):
        nltk.download('stopwords')
        stop_words = stopwords.words('english')
        vectorizer = CountVectorizer(stop_words=stop_words, max_features=1000)
        X = vectorizer.fit_transform(self.df['headline'].fillna(''))
        keywords = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        top_words = keywords.sum().sort_values(ascending=False).head(20)
        top_words.plot(kind='barh', title="Top Keywords in Headlines")
        plt.gca().invert_yaxis()
        plt.xlabel("Frequency")
        plt.tight_layout()
        plt.show()

    def extract_email_domains(self):
        self.df['publisher_domain'] = self.df['publisher'].apply(
            lambda pub: re.search(r'@([A-Za-z0-9.-]+\.[A-Za-z]{2,})', str(pub)).group(1)
            if re.search(r'@([A-Za-z0-9.-]+\.[A-Za-z]{2,})', str(pub)) else None
        )
        email_domains = self.df['publisher_domain'].dropna()
        domain_counts = email_domains.value_counts()
        if not domain_counts.empty:
            print("Top contributing email domains:")
            print(domain_counts.head(10))
            plt.figure(figsize=(10, 6))
            domain_counts.head(10).plot(kind='barh', color='teal')
            plt.title("Top Email Domains in Publisher Names")
            plt.xlabel("Number of Articles")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
        else:
            print("No email-format publishers were found.")
