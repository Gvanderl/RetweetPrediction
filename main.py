from utils import models, dataprocessor
import config
from sklearn.model_selection import train_test_split
import pandas as pd


if __name__ == "__main__":
    dp = dataprocessor.DataProcessor()
    dp.load_csv()
    X, y = dp.get_numerical()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = models.LinearForest()

    model.load_or_train(X_train, y_train)
    model.evaluate(X_test, y_test)
    model.save_models()

    dp_val = dataprocessor.DataProcessor()
    dp_val.load_csv(config.eval_path)
    X, _ = dp_val.get_numerical()
    our = pd.DataFrame(model.predict(X))
    our.index = X.index
    our.columns = ["NoRetweets"]
    our.to_csv(config.data_folder / "predictions.csv", index_label="TweetID")



