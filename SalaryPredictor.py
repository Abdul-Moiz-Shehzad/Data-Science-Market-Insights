import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

class SalaryPredictor:
    def __init__(self):
        self.scalers = {}
        self.models = {}
        self.feature_names = None
    def get_model(self, model_name):
        """Retrieve the trained model by name."""
        if model_name in self.models:
            return self.models[model_name]
        else:
            raise ValueError(f"Model {model_name} not found.")
    def load_and_preprocess(self, df):
        try:
            if 'Unnamed: 0' in df.columns:
                df.drop(
                    ['Unnamed: 0', 'Salary Estimate', 'Job Description', 
                     'Location', 'Founded', 'URL', 'Job Title', 'Minimum Salary'], 
                    axis=1, inplace=True
                )

            selected_values = [-1, '-1', 'Unknown / Non-Applicable', '--']
            numerical_features = ['Company Age', 'Rating']

            self.scalers['numerical'] = MinMaxScaler()
            df[numerical_features] = self.scalers['numerical'].fit_transform(df[numerical_features])

            self.scalers['target'] = MinMaxScaler()
            df['Average Salary'] = self.scalers['target'].fit_transform(df[['Average Salary']])

            df_encoded = pd.get_dummies(df)

            self.feature_names = [col for col in df_encoded.columns if col != 'Average Salary']
            X = df_encoded[self.feature_names]
            y = df_encoded['Average Salary']

            return train_test_split(X, y, test_size=0.3, random_state=42)
        except Exception as e:
            print(f"Error during data preprocessing: {e}")
            raise


    def train_ridge(self, X_train, X_test, y_train, y_test):
        params = {
            'alpha': [0.01, 0.1, 1.0, 10.0],
            'solver': ['auto', 'svd', 'cholesky']
        }
        ridge = Ridge()
        grid_search = GridSearchCV(ridge, params, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        self.models['ridge'] = grid_search.best_estimator_
        return self._evaluate_model('ridge', X_test, y_test)

    def train_linear_regression(self, X_train, X_test, y_train, y_test):
        lr=LinearRegression()
        lr.fit(X_train,y_train)
        self.models['linear']=lr
        return self._evaluate_model('linear',X_test,y_test)

    def train_lasso(self, X_train, X_test, y_train, y_test):
        params = {
            'alpha': [0.001, 0.01, 0.1, 1.0],
            'selection': ['cyclic', 'random']
        }
        lasso = Lasso()
        grid_search = GridSearchCV(lasso, params, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        self.models['lasso'] = grid_search.best_estimator_

        feature_importance = pd.Series(
            grid_search.best_estimator_.coef_,
            index=self.feature_names
        )
        important_features = feature_importance[feature_importance != 0].index

        return important_features, self._evaluate_model('lasso', X_test, y_test)

    def train_random_forest(self, X_train, X_test, y_train, y_test):
        params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        rf = RandomForestRegressor(random_state=42)
        grid_search = RandomizedSearchCV(rf, params, n_iter=20, cv=5,
                                       scoring='neg_mean_squared_error', random_state=42)
        grid_search.fit(X_train, y_train)

        self.models['rf'] = grid_search.best_estimator_
        return self._evaluate_model('rf', X_test, y_test)

    def train_gradient_boost(self, X_train, X_test, y_train, y_test):
        params = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
        gboost = GradientBoostingRegressor(random_state=42)
        grid_search = GridSearchCV(gboost, params, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        self.models['gboost'] = grid_search.best_estimator_
        return self._evaluate_model('gboost', X_test, y_test)



    def _evaluate_model(self, model_name, X_test, y_test):
        predictions = self.models[model_name].predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        predictions_original = self.scalers['target'].inverse_transform(
            predictions.reshape(-1, 1)
        ).flatten()

        return {
            'rmse': rmse,
            'r2': r2,
            'predictions': predictions_original
        }

    def predict(self, X_new, model_name='xgb'):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")

        predictions = self.models[model_name].predict(X_new)
        return self.scalers['target'].inverse_transform(
            predictions.reshape(-1, 1)
        ).flatten()

    def evaluate_all_models(self, X_test, y_test):
        results = []

        for model_name, model in self.models.items():
            predictions = model.predict(X_test)

            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)

            predictions_original = self.scalers['target'].inverse_transform(
                predictions.reshape(-1, 1)
            ).flatten()
            y_test_original = self.scalers['target'].inverse_transform(
                y_test.values.reshape(-1, 1)
            ).flatten()

            mse_original = mean_squared_error(y_test_original, predictions_original)
            rmse_original = np.sqrt(mse_original)
            results.append({
            'Model': model_name,
            'RMSE (Scaled)': rmse,
            'R² (Scaled)': r2,
            'RMSE (Original)': rmse_original
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='RMSE (Scaled)').reset_index(drop=True)

        return results_df