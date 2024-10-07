import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Class to handle Model Training, Tuning, and Evaluation
class ModelPipeline:
    def __init__(self, data, target_column):
        """
        Initialize the pipeline with data and target column
        """
        self.data = data
        self.target_column = target_column
        self.X = data.drop(columns=[target_column])
        self.y = data[target_column]
        self.models = {}
        self.results = {}
        self.preprocessor = None

    def preprocess_data(self):
        """
        Preprocess the data: handle missing values, scale numeric features, and encode categorical features
        """
        numeric_features = self.X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.X.select_dtypes(include=['object']).columns
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])
        
        self.X = self.preprocessor.fit_transform(self.X)
        print(f"Data preprocessing completed: Shape of processed data = {self.X.shape}")

    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the dataset into training and testing sets
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        print(f"Data split completed: Training data size = {self.X_train.shape}, Test data size = {self.X_test.shape}")

    def add_model(self, model_name, model):
        """
        Add models to the pipeline for training
        """
        self.models[model_name] = model

    # def train_model(self, model_name):
    #     """
    #     Train a model and ensure proper handling of model attributes
    #     """
    #     model = self.models.get(model_name)
    #     if model:
    #         try:
    #             # Train the model
    #             model.fit(self.X_train, self.y_train)
                
    #             # Ensure the model is trained successfully by accessing key attributes
    #             if hasattr(model, 'estimators_'):
    #                 print(f"Model {model_name} trained successfully with {len(model.estimators_)} estimators.")
    #             else:
    #                 print(f"Model {model_name} trained successfully. No 'estimators_' attribute (non-ensemble model).")
            
    #         except AttributeError as ae:
    #             print(f"AttributeError: {ae}. This model might not have 'estimators_' (e.g., Logistic Regression).")
    #         except Exception as e:
    #             print(f"Error while training {model_name}: {e}")
    #     else:
    #         raise ValueError(f"Model {model_name} not found in pipeline.")

    def train_models(self):
        # Logistic Regression
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(random_state=42)
        lr.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = lr

        # Decision Tree
        from sklearn.tree import DecisionTreeClassifier
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(self.X_train, self.y_train)
        self.models['Decision Tree'] = dt

        # Random Forest
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(random_state=42)
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf

        # Gradient Boosting Machine
        from sklearn.ensemble import GradientBoostingClassifier
        gbm = GradientBoostingClassifier(random_state=42)
        gbm.fit(self.X_train, self.y_train)
        self.models['Gradient Boosting'] = gbm


    def hyperparameter_tuning(self, model_name, param_grid, search_type='grid', cv=5):
        """
        Perform hyperparameter tuning using Grid Search or Random Search
        """
        model = self.models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found in pipeline.")
        
        if search_type == 'grid':
            search = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1, verbose=1)
        elif search_type == 'random':
            search = RandomizedSearchCV(model, param_grid, cv=cv, n_jobs=-1, verbose=1)
        else:
            raise ValueError("Search type must be 'grid' or 'random'.")
        
        search.fit(self.X_train, self.y_train)
        self.models[model_name] = search.best_estimator_
        print(f"Best hyperparameters for {model_name}: {search.best_params_}")
        
    def evaluate_model(self, model_name):
        """
        Evaluate the model on test data
        """
        model = self.models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found in pipeline.")
        
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        self.results[model_name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC-AUC": roc_auc
        }
        
        print(f"Evaluation metrics for {model_name}:")
        for metric, value in self.results[model_name].items():
            print(f"{metric}: {value:.4f}")
    
    def plot_roc_curve(self, model_name):
        """
        Plot ROC curve for a model
        """
        model = self.models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found in pipeline.")
        
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(self.y_test, y_pred_proba):.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label="Random guess (AUC = 0.5)")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {model_name}')
        plt.legend()
#         plt.show()

# # Example usage:
# # Assuming data is pre-loaded into 'data' and the target column is defined as 'target_column'
# # pipeline = ModelPipeline(data, target_column)

# Preprocess the data
# pipeline.preprocess_data()

# # Split the data
# # pipeline.split_data()

# # Add models to the pipeline
# pipeline.add_model('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42))
# pipeline.add_model('Random Forest', RandomForestClassifier(random_state=42))

# # Train the models
# pipeline.train_model('Logistic Regression')
# pipeline.train_model('Random Forest')

# # Hyperparameter tuning
# param_grid_rf = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
# }
# pipeline.hyperparameter_tuning('Random Forest', param_grid_rf, search_type='grid')

# # Evaluate models
# pipeline.evaluate_model('Logistic Regression')
# pipeline.evaluate_model('Random Forest')

# # Plot ROC Curve
# pipeline.plot_roc_curve('Random Forest')
