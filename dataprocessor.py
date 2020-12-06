from config import *
import gensim
import pandas as pd
import os
import re
import string as st
import numpy as np
from verstack.stratified_continuous_split import scsplit


class DataProcessor:
    def __init__(self):
        self.df = None
        self.w2v_model = None
        self.label_col = "retweet_count"
        self.label_max = None

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
        return scsplit(self.df["text"], self.df[self.label_col], stratify=self.df[self.label_col], train_size=0.7,
                       test_size=0.3)

    def apply_glove(self):
        print("Loading Glove ...")
        self.get_glove_model()
        print("Done")
        print("Processing data ...")
        self.load_csv(train_path)
        self.clean_df()
        self.norm_label()
        self.df = self.df.replace('', np.nan).dropna(subset=["text"]).reset_index(drop=True)
        self.df["text"] = self.df["text"].apply(self.tweet_w2v)
        print("Done")

    def save_df(self, path):
        self.df.to_hdf(path, key='df', mode='w')
        print(f"Dataframe saved to {path}")

    def load_csv(self, path, nrows=None):
        self.df = pd.read_csv(path, nrows=nrows)


if __name__ == '__main__':
    data_processor = DataProcessor()
    data_processor.apply_glove()
    data_processor.save_df(data_folder / "glove_prepro.h5")
    X_train, X_test, y_train, y_test = data_processor.get_split_df()
    print(f"X_train = {X_train}")
    print(f"y_train = {y_train}")
