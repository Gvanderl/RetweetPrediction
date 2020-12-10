from config import *
import gensim
import pandas as pd
import os
import re
import string as st
import numpy as np
from textblob import TextBlob


class DataProcessor:
    def __init__(self, nrows=None):
        self.df = None
        self.w2v_model = None
        self.label_col = "retweet_count"
        self.label_max = None
        self.drop = False
        self.nrows = nrows

    def get_glove_model(self):
        if not glove_path.exists():
            os.system(f"wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz "
                      f"--directory-prefix={data_folder}")
        self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format(glove_path, binary=True)

    def tweet_w2v(self, text):
        vecs = np.array([self.w2v_model[word] for word in text.split() if word in self.w2v_model])
        return np.mean(vecs, axis=0)

    def clean_df(self):
        self.df["text"] = self.df["text"].apply(self.clean)
        if self.drop:
            self.df = self.df.replace('', np.nan).dropna(subset=["text"]).reset_index(drop=True)

    def clean(self, string):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)

        punc = (st.punctuation.replace('@', '').replace('#', '')) + '"' + "'" + '”' + '“' + '‘'
        translator = str.maketrans('', '', punc)
        string = str(string).lower()
        string = string.translate(translator)
        string = string.split()
        to_remove = []
        for word in string:
            if word[0] == '#' or word[0] == '@' or word == 'rt' or word[:4] == 'http' or word[0].isnumeric():
                to_remove.append(word)
        for word in to_remove:
            string.remove(word)
        text = emoji_pattern.sub(r'', ' '.join(string))
        text = re.sub("[^a-zA-Z ]", "", text)
        return text

    def norm_label(self, overwrite_max=None):
        if overwrite_max:
            self.label_max = overwrite_max
        else:
            self.label_max = max(self.df[self.label_col])
        self.df[self.label_col] = self.df[self.label_col] / self.label_max

    def unnorm(self, norm_labels):
        return norm_labels * self.label_max

    def get_split_df(self):
        from verstack.stratified_continuous_split import scsplit
        return scsplit(self.df["text"], self.df[self.label_col], stratify=self.df[self.label_col], train_size=0.7,
                       test_size=0.3)

    def get_split_df_with_all_cols(self):
        from verstack.stratified_continuous_split import scsplit
        return scsplit(self.df, self.df[self.label_col], stratify=self.df[self.label_col], train_size=0.7,
                       test_size=0.3)

    def apply_glove(self, normalize=True):
        print("Loading Glove ...")
        self.get_glove_model()
        print("Done")
        print("Processing data ...")
        self.clean_df()
        if normalize:
            self.norm_label()
        if self.drop:
            self.df = self.df.replace('', np.nan).dropna(subset=["text"]).reset_index(drop=True)
        self.df["text"] = self.df["text"].apply(self.tweet_w2v)
        print("Done")

    def save_df(self, path):
        self.df.to_hdf(path, key='df', mode='w')
        print(f"Dataframe saved to {path}")

    def load_csv(self, path=train_path):
        self.df = pd.read_csv(path, nrows=self.nrows, index_col="id")
        self.df["user_verified"] = self.df["user_verified"].astype(bool)

    def replace_timestamp(self):
        self.df["day_of_week"] = pd.to_datetime(self.df["timestamp"]).dt.weekday
        self.df["hour"] = pd.to_datetime(self.df["timestamp"]).dt.hour
        self.df.drop("timestamp", 1, inplace=True)

    def add_sentiment(self):
        print("Doing sentiment analysis...")
        self.df['positivity'] = self.df["text"].apply(lambda x: max(TextBlob(x).sentiment.polarity, 0))
        self.df['negativity'] = self.df["text"].apply(lambda x: -min(TextBlob(x).sentiment.polarity, 0))
        self.df['subjectivity'] = self.df["text"].apply(lambda x: TextBlob(x).sentiment.subjectivity)
        print("Done")

    def filter_df(self):
        self.clean_df()
        self.add_sentiment()
        self.replace_timestamp()
        # self.df = self.df[self.df["user_statuses_count"] > 10]

    def get_numerical(self):
        self.filter_df()
        to_drop = ["urls", "hashtags", "text", "user_mentions"]
        dummies_cols = ["day_of_week", "hour"]
        self.df = pd.get_dummies(self.df, columns=dummies_cols)
        if self.label_col in self.df.columns:
            to_drop.append(self.label_col)
            y_train = self.df[self.label_col]
            y_train = np.log1p(y_train.astype(float))
        else:
            y_train = None
        features = self.df.columns.drop(to_drop)
        X_train = self.df[features]

        return np.log1p(X_train.astype(float)), y_train


if __name__ == '__main__':
    data_processor = DataProcessor()
    data_processor.load_csv()
    data_processor.apply_glove(False)
    data_processor.save_df(data_folder / "glove_prepro.h5")
    X_train, X_test, y_train, y_test = data_processor.get_split_df()
    print(X_train.shape)
    print(f"X_train = {X_train}")
    print(f"y_train = {y_train}")
