from config import *
import gensim
import pandas as pd
import os
import re
import string as st
import numpy as np
from verstack.stratified_continuous_split import scsplit


def get_GloVe():
    if not glove_path.exists():
        os.system(f"wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz "
                  f"--directory-prefix={data_folder}")
    return gensim.models.KeyedVectors.load_word2vec_format(glove_path, binary=True)


def process_tweet(text, w2v_model):
    vecs = np.array([w2v_model[word] for word in text.split() if word in w2v_model])
    return np.mean(vecs, axis=0)


def preprocess_data(df, w2v_model):
    df["text"] = df["text"].apply(clean)
    df = df.replace('', np.nan).dropna(subset=["text"]).reset_index()
    x = df["text"].apply(lambda x: process_tweet(x, w2v_model))
    y = df["retweet_count"]
    return x, y


def clean(string):
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


def get_data():
    print("Loading Glove ...")
    glove_model = get_GloVe()
    print("Done")
    train_df = pd.read_csv(train_path, index_col=0)
    print("Processing data ...")
    x, y = preprocess_data(train_df, glove_model)
    print("Done")
    return scsplit(x, y, stratify=y, train_size=0.7, test_size=0.3)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()
    np.savez(data_folder / "glove_prepro", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    print(f"X_train = {X_train}")
    print(f"y_train = {y_train}")
