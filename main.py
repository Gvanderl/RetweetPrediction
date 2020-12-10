from utils import models, dataprocessor
import config
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # Load dataset for training
    clean_df_path = config.data_folder / "clean_train.csv"
    dp = dataprocessor.DataProcessor()
    if clean_df_path.exists():
        dp.load_csv(clean_df_path)
    else:
        dp.load_csv()
        dp.filter_df()
        dp.df.to_csv(clean_df_path)
    X, y = dp.get_numerical(False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Get trained model
    model = models.LinearForest()
    model.load_or_train(X_train, y_train)
    model.evaluate(X_test, y_test)

    # Get validation data
    dp_val = dataprocessor.DataProcessor()
    dp_val.load_csv(config.eval_path)
    X, _ = dp_val.get_numerical()

    # Predict validation data
    our = pd.DataFrame(np.maximum(np.exp(model.predict(X)).round().astype(int) - 1, 0))
    our.index = X.index
    our.columns = ["NoRetweets"]

    # Save predictions to file
    pred_path = config.output_folder / "predictions.csv"
    our.to_csv(pred_path, index_label="TweetID")
    print(f"Saved predictions at {pred_path}")
