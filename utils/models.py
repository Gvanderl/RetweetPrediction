from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pickle
from config import *
import numpy as np
from sklearn.metrics import mean_absolute_error

class LinearForest:
    def __init__(self, lr_config=None, rf_config=None):
        if lr_config is None:
            lr_config = {
                "parameters": {
                    "fit_intercept": False
                },
                "file_name": "lr.pkl"
            }
        if rf_config is None:
            rf_config = {
                "parameters": {
                    "max_depth": 20,
                    "n_estimators": 500,
                    "random_state": 7,
                    "n_jobs": 3,
                    "verbose": 5
                },
                "file_name": "rf_20d_500e_3j.pkl"
            }
        self.lr_config = lr_config
        self.rf_config = rf_config
        self.lr_model = None
        self.rf_model = None

    def train_models(self, X_train, y_train):
        # Train Linear Regression model
        self.lr_model = LinearRegression(**self.lr_config["parameters"])
        self.lr_model.fit(X_train, y_train)

        # Train Random forest model
        y_train_rf = self.lr_model.predict(X_train)
        self.rf_model = RandomForestRegressor(**self.rf_config["parameters"])
        self.rf_model.fit(X_train, y_train_rf)

    def save_models(self):
        pickle.dump(self.lr_model, open(model_folder / self.lr_config["file_name"], "wb"))
        pickle.dump(self.rf_model, open(model_folder / self.rf_config["file_name"], "wb"))

    def load_models(self):
        self.lr_model = pickle.load(open(model_folder / self.lr_config["file_name"], "rb"))
        self.rf_model = pickle.load(open(model_folder / self.rf_config["file_name"], "rb"))

    def predict(self, X):
        assert self.lr_model is not None
        assert self.rf_model is not None
        lr_pred = self.lr_model.predict(X)
        rf_pred = self.rf_model.predict(X)
        return self.get_normal_counter(lr_pred, rf_pred)

    def get_normal_counter(self, n, logarithm="10"):
        return np.array([max(0, x) for x in (np.exp(n).round().astype(int) - 1)])

    def load_or_train(self, X_train, y_train):
        if (model_folder / self.lr_config["file_name"]).exists() and (model_folder / self.rf_config["file_name"]).exists():
            self.load_models()
        else:
            self.train_models(X_train, y_train)

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        print(f"MSE is {mean_absolute_error(y_true=y_test, y_pred=preds):.2f}")
