import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class EDA:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the EDA class with the dataset.
        """
        self.data = data
    
    def dataset_overview(self):
        """
        Returns an overview of the dataset including the number of rows, columns, and data types.
        """
        print("Dataset Overview:")
        print(f"Number of Rows: {self.data.shape[0]}")
        print(f"Number of Columns: {self.data.shape[1]}")
        print("\nData Types:")
        print(self.data.dtypes)
        print("\nMissing Values Count:")
        print(self.data.isnull().sum())
    
    def summary_statistics(self):
        """
        Returns summary statistics for numerical features in the dataset.
        """
        print("Summary Statistics for Numerical Features:")
        return self.data.describe()
    
    def numerical_distribution(self):
        """
        Visualizes the distribution of numerical features using histograms and/or KDE plots.
        """
        num_features = self.data.select_dtypes(include=['int64', 'float64']).columns
        for feature in num_features:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.data[feature], kde=True, bins=30)
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.show()
    
    def categorical_distribution(self):
        """
        Visualizes the distribution of categorical features using bar plots.
        """
        cat_features = self.data.select_dtypes(include=['object', 'category']).columns
        for feature in cat_features:
            plt.figure(figsize=(10, 6))
            sns.countplot(x=self.data[feature], order=self.data[feature].value_counts().index)
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.show()

    def correlation_analysis(self):
        """
        Displays the correlation matrix of numerical features and visualizes it using a heatmap.
        """
        plt.figure(figsize=(10, 8))
        corr_matrix = self.data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.show()

    def identify_missing_values(self):
        """
        Identifies missing values and calculates the percentage of missing data for each feature.
        """
        missing_data = self.data.isnull().sum()
        missing_percent = (missing_data / len(self.data)) * 100
        missing_report = pd.DataFrame({'Missing Values': missing_data, 'Percentage': missing_percent})
        missing_report = missing_report[missing_report['Missing Values'] > 0].sort_values(by='Percentage', ascending=False)
        print("Missing Values Report:")
        print(missing_report)
        return missing_report

    def outlier_detection(self):
        """
        Detects outliers in numerical features using box plots.
        """
        num_features = self.data.select_dtypes(include=['int64', 'float64']).columns
        for feature in num_features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=self.data[feature])
            plt.title(f'Outlier Detection for {feature}')
            plt.xlabel(feature)
            plt.show()
