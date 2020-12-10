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

    def train_models(self, X_train, y_train, save=True):
        # Train Linear Regression model
        X_train_log = np.log1p(X_train.astype(float))
        y_train_log = np.log1p(y_train.astype(float))
        self.lr_model = LinearRegression(**self.lr_config["parameters"])
        self.lr_model.fit(X_train_log, y_train_log)

        # Train Random forest model
        y_train_rf = y_train_log - self.lr_model.predict(X_train_log)
        self.rf_model = RandomForestRegressor(**self.rf_config["parameters"])
        self.rf_model.fit(X_train, y_train_rf)

        if save:
            self.save_models()

    def save_models(self):
        pickle.dump(self.lr_model, open(model_folder / self.lr_config["file_name"], "wb"))
        pickle.dump(self.rf_model, open(model_folder / self.rf_config["file_name"], "wb"))
        print("Saved models")

    def load_models(self):
        self.lr_model = pickle.load(open(model_folder / self.lr_config["file_name"], "rb"))
        self.rf_model = pickle.load(open(model_folder / self.rf_config["file_name"], "rb"))
        print("Loaded models")

    def predict(self, X):
        assert self.lr_model is not None, "You must first load or train the linear regression model"
        assert self.rf_model is not None, "You must first load or train the random forest model"
        X_log = np.log1p(X.astype(float))
        lr_pred = self.lr_model.predict(X_log)
        rf_pred = self.rf_model.predict(X)
        return lr_pred + rf_pred

    def load_or_train(self, X_train, y_train):
        if (model_folder / self.lr_config["file_name"]).exists() and (model_folder / self.rf_config["file_name"]).exists():
            self.load_models()
        else:
            self.train_models(X_train, y_train)

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        preds = np.maximum(np.exp(preds).round().astype(int) - 1, 0)
        print(f"Testing MSE is {mean_absolute_error(y_true=y_test, y_pred=preds):.2f}")
