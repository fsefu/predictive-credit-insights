import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
# from xverse import Xverse
# from woe import WOE

# 1. Aggregate Features Class

class AggregateFeatures:
    def __init__(self, data: pd.DataFrame, customer_id_col: str, transaction_amount_col: str):
        self.data = data
        self.customer_id_col = customer_id_col
        self.transaction_amount_col = transaction_amount_col

    def compute_aggregate_features(self) -> pd.DataFrame:
        agg_data = self.data.groupby(self.customer_id_col).agg(
            total_transaction_amount=(self.transaction_amount_col, 'sum'),
            average_transaction_amount=(self.transaction_amount_col, 'mean'),
            transaction_count=(self.transaction_amount_col, 'count'),
            std_dev_transaction_amount=(self.transaction_amount_col, 'std'),
        ).reset_index()
        # Fill NaN values in std_dev_transaction_amount with 0 (single transaction has 0 variability)
        agg_data['std_dev_transaction_amount'] = agg_data['std_dev_transaction_amount'].fillna(0)

        return agg_data

    def extract_features(self) -> pd.DataFrame:
        self.data['transaction_hour'] = self.data['transaction_date'].dt.hour
        self.data['transaction_day'] = self.data['transaction_date'].dt.day
        self.data['transaction_month'] = self.data['transaction_date'].dt.month
        self.data['transaction_year'] = self.data['transaction_date'].dt.year
        return self.data

    def encode_categorical(self, categorical_cols: list) -> pd.DataFrame:
        for col in categorical_cols:
            one_hot = pd.get_dummies(self.data[col], prefix=col)
            self.data = pd.concat([self.data, one_hot], axis=1)
            self.data.drop(col, axis=1, inplace=True)
        return self.data

    def handle_missing_values(self) -> pd.DataFrame:
        imputer = SimpleImputer(strategy='mean')
        self.data[[self.transaction_amount_col]] = imputer.fit_transform(self.data[[self.transaction_amount_col]])
        return self.data

    def normalize_numerical_features(self, numerical_cols: list) -> pd.DataFrame:
        scaler = MinMaxScaler()
        self.data[numerical_cols] = scaler.fit_transform(self.data[numerical_cols])
        return self.data

    def calculate_woe_iv(self, target_col: str, categorical_cols: list) -> pd.DataFrame:
        woe_iv_df = pd.DataFrame(columns=['Feature', 'WoE', 'IV'])

        for col in categorical_cols:
            total_events = self.data[target_col].sum()
            total_non_events = self.data[target_col].count() - total_events
            
            grouped = self.data.groupby(col)[target_col].agg(['count', 'sum']).reset_index()
            grouped.columns = [col, 'total_count', 'event_count']
            grouped['non_event_count'] = grouped['total_count'] - grouped['event_count']

            grouped['event_rate'] = grouped['event_count'] / total_events
            grouped['non_event_rate'] = grouped['non_event_count'] / total_non_events
            grouped['WoE'] = np.log(grouped['event_rate'] / grouped['non_event_rate'])

            grouped['IV'] = (grouped['event_rate'] - grouped['non_event_rate']) * grouped['WoE']
            woe = grouped['WoE'].sum()
            iv = grouped['IV'].sum()

            woe_iv_df = woe_iv_df.append({'Feature': col, 'WoE': woe, 'IV': iv}, ignore_index=True)

        return woe_iv_df

# 2. Time Feature Extraction Class
class TimeFeatureExtractor:
    def __init__(self, data: pd.DataFrame, datetime_col: str):
        self.data = data
        self.datetime_col = datetime_col

    def extract_time_features(self) -> pd.DataFrame:
        self.data['transaction_hour'] = pd.to_datetime(self.data[self.datetime_col]).dt.hour
        self.data['transaction_day'] = pd.to_datetime(self.data[self.datetime_col]).dt.day
        self.data['transaction_month'] = pd.to_datetime(self.data[self.datetime_col]).dt.month
        self.data['transaction_year'] = pd.to_datetime(self.data[self.datetime_col]).dt.year
        return self.data

# 3. Categorical Encoding Class
class CategoricalEncoder:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def one_hot_encode(self, columns: list) -> pd.DataFrame:
        # Ensure columns exist in the DataFrame
        if not all(col in self.data.columns for col in columns):
            raise ValueError("One or more columns are not present in the DataFrame.")
        
        # Initialize OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False, drop=None)  # Keep all categories
        encoded_data = encoder.fit_transform(self.data[columns])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(columns), index=self.data.index)
        
        # Concatenate the new encoded columns with original data and drop the original columns
        transformed_data = pd.concat([self.data.drop(columns, axis=1), encoded_df], axis=1)
        return transformed_data

    def label_encode(self, columns: list) -> pd.DataFrame:
        # Ensure columns exist in the DataFrame
        if not all(col in self.data.columns for col in columns):
            raise ValueError("One or more columns are not present in the DataFrame.")
        
        transformed_data = self.data.copy()
        # Apply label encoding to each column
        for col in columns:
            encoder = LabelEncoder()
            transformed_data[col] = encoder.fit_transform(transformed_data[col].astype(str))  # Convert to str for safety
        return transformed_data

# 4. Missing Value Handler Class
class MissingValueHandler:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def impute_missing_values(self, numerical_columns: list = None, categorical_columns: list = None) -> pd.DataFrame:
        # Impute numerical columns with mean/median
        if numerical_columns:
            num_imputer = SimpleImputer(strategy='mean')  # Can be changed to 'median' based on preference
            self.data[numerical_columns] = num_imputer.fit_transform(self.data[numerical_columns])
        
        # Impute categorical columns with most frequent value
        if categorical_columns:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            self.data[categorical_columns] = cat_imputer.fit_transform(self.data[categorical_columns])
        
        return self.data

    def remove_missing_values(self, threshold: float = 0.05) -> pd.DataFrame:
        missing_percent = self.data.isnull().mean()
        columns_to_remove = missing_percent[missing_percent > threshold].index
        
        # Log or display the columns being removed
        if len(columns_to_remove) > 0:
            print(f"Removing columns with more than {threshold*100}% missing values: {list(columns_to_remove)}")
        
        self.data = self.data.drop(columns=columns_to_remove)
        return self.data

    def save_cleaned_data(self, filename: str = "cleaned_data.csv") -> None:
        """Save the cleaned data to a CSV file."""
        self.data.to_csv(filename, index=False)
        print(f"Cleaned data saved to {filename}")
        
# 5. Normalization and Standardization Class
class Scaler:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def normalize(self, columns: list) -> pd.DataFrame:
        scaler = MinMaxScaler()
        self.data[columns] = scaler.fit_transform(self.data[columns])
        return self.data

    def standardize(self, columns: list) -> pd.DataFrame:
        scaler = StandardScaler()
        self.data[columns] = scaler.fit_transform(self.data[columns])
        return self.data

# 6. WOE/IV Feature Engineering Class
class WOEIVFeatureEngineering:
    def __init__(self, data: pd.DataFrame, target_col: str):
        self.data = data
        self.target_col = target_col

    def calculate_woe_iv(self, feature_cols: list) -> pd.DataFrame:
        woe_iv_df = pd.DataFrame(columns=['Feature', 'WoE', 'IV'])

        total_events = self.data[self.target_col].sum()
        total_non_events = self.data[self.target_col].count() - total_events

        for col in feature_cols:
            # Group the data by the feature column and calculate counts
            grouped = self.data.groupby(col)[self.target_col].agg(['count', 'sum']).reset_index()
            grouped.columns = [col, 'total_count', 'event_count']
            grouped['non_event_count'] = grouped['total_count'] - grouped['event_count']

            # Calculate event and non-event rates
            grouped['event_rate'] = grouped['event_count'] / total_events
            grouped['non_event_rate'] = grouped['non_event_count'] / total_non_events

            # Calculate WoE and IV
            grouped['WoE'] = np.log(grouped['event_rate'] / grouped['non_event_rate'].replace(0, np.nan))
            grouped['IV'] = (grouped['event_rate'] - grouped['non_event_rate']) * grouped['WoE']

            # Aggregate WoE and IV values for the feature
            woe = grouped['WoE'].sum()
            iv = grouped['IV'].sum()

            # Append the results to the output DataFrame
            woe_iv_df = woe_iv_df.append({'Feature': col, 'WoE': woe, 'IV': iv}, ignore_index=True)

        return woe_iv_df
