from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

class CreditRiskModel:
    def __init__(self, data, target_column):
        self.data = data
        self.target_column = target_column
        self.models = {}
    
    def preprocess_data(self):
        # Drop unnecessary columns including TransactionStartTime
        drop_columns = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'TransactionStartTime']
        self.data = self.data.drop(columns=drop_columns)
        
        # Convert categorical features to numerical (using OneHot or Label Encoding)
        categorical_columns = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId']
        self.data = pd.get_dummies(self.data, columns=categorical_columns, drop_first=True)

        # Ensure FraudResult is numeric
        self.data[self.target_column] = self.data[self.target_column].astype(int)

        # Split features and target
        self.X = self.data.drop(columns=[self.target_column])
        self.y = self.data[self.target_column]

        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)   
         
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
    
    def hyperparameter_tuning(self, model_name, param_grid):
        from sklearn.model_selection import GridSearchCV
        model = self.models[model_name]
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
        grid_search.fit(self.X_train, self.y_train)
        self.models[model_name] = grid_search.best_estimator_
    
    def evaluate_all_models(self):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        evaluation_results = {}
        for model_name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            evaluation_results[model_name] = {
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred),
                'Recall': recall_score(self.y_test, y_pred),
                'F1 Score': f1_score(self.y_test, y_pred),
                'ROC-AUC': roc_auc_score(self.y_test, y_pred)
            }
        
        return evaluation_results
