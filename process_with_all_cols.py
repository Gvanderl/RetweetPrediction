import pandas as pd
import logging
from dataprocessor import DataProcessor
from config import *

data_processor = DataProcessor()
data_processor.apply_glove(False, 1000)

print("shape = ", data_processor.df.shape)

X_train, X_test, y_train, y_test = data_processor.get_split_df_with_all_cols()

train_df = pd.DataFrame({"timestamp": X_train["timestamp"], "user_verified": X_train["user_verified"], "user_statuses_count": X_train["user_statuses_count"], "user_followers_count": X_train["user_followers_count"], "user_friends_count": X_train["user_friends_count"],  "text": X_train["text"]})
train_labels = pd.DataFrame({"retweet_count": y_train})
eval_df = pd.DataFrame({"timestamp": X_test["timestamp"], "user_verified": X_test["user_verified"], "user_statuses_count": X_test["user_statuses_count"], "user_followers_count": X_test["user_followers_count"], "user_friends_count": X_test["user_friends_count"],  "text": X_test["text"]})
eval_labels = pd.DataFrame({"retweet_count": y_test})

print(train_df.shape)
print(train_df["text"][0])

train_df.to_hdf(data_folder / "train.h5", "train")
eval_df.to_hdf(data_folder / "eval.h5", "eval")
train_labels.to_hdf(data_folder / "train_labels.h5", "eval")
eval_labels.to_hdf(data_folder / "eval_labels.h5", "eval")

print("save done")
