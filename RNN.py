from config import *
import numpy as np
import csv
import pandas as pd
import time #pour nommer les models
from keras.models import Sequential
from keras.layers import Dense, Activation, Input, Add, AveragePooling2D
from keras.layers import Conv1D, Conv2D, ReLU, BatchNormalization, Flatten
from keras.models import Model
from tensorflow import Tensor

batch_size = 128
epochs = 10
n = 3
depth = n*6+2
model_name = "ResNet-%d" %depth + time.strftime("%Y%m%d-%H%M%S")
lr = 1e-5
num_classes = 4

def get_data():
    train_df = pd.read_hdf(data_folder / "train.h5")
    eval_df = pd.read_hdf(data_folder / "eval.h5")
    train_labels = pd.read_hdf(data_folder / "train_labels.h5")
    eval_labels = pd.read_hdf(data_folder / "eval_labels.h5")

    train_df = train_df.to_numpy()
    eval_df = eval_df.to_numpy()
    train_labels = train_labels.to_numpy()
    eval_labels = eval_labels.to_numpy()
    
    res_train_df =np.zeros((train_df.shape[0], train_df.shape[1]-1+train_df[0][train_df.shape[1]-1].shape[0]))

    res_train_df[:, 0:train_df.shape[1]-1] = train_df[:, 0:train_df.shape[1]-1]
    j = train_df.shape[1]-1
    for i in range(train_df.shape[0]):
        for x in range(train_df[0][train_df.shape[1]-1].shape[0]):
            if not np.isnan(train_df[i][j]).any():
                res_train_df[i][j+x] = train_df[i][j][x]
    
    res_eval_df =np.zeros((eval_df.shape[0], eval_df.shape[1]-1+eval_df[0][eval_df.shape[1]-1].shape[0]))

    res_eval_df[:, 0:eval_df.shape[1]-1] = eval_df[:, 0:eval_df.shape[1]-1]
    j = eval_df.shape[1]-1
    for i in range(eval_df.shape[0]):
        for x in range(eval_df[0][eval_df.shape[1]-1].shape[0]):
            if not np.isnan(eval_df[i][j]).any():
                res_eval_df[i][j+x] = eval_df[i][j][x]
                
    return res_train_df, res_eval_df, train_labels, eval_labels

def add_block(inputs):
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

def residual_block(x, downsample, filters, kernel_size = 2):
    y = Conv1D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = add_block(y)
    y = Conv1D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv1D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = add_block(out)
    return out


def create_res_net(x, y, num_filters = 16):
    inputs = Input(shape=(x.shape[1], 1))
    t = BatchNormalization()(inputs)
    t = Conv1D(kernel_size=2,
               strides=1,
               filters=num_filters,
               padding="same")(inputs)
    t = add_block(t)

    num_blocks_list = [2, 5, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j == 0 and i != 0), filters=num_filters)
        num_filters *= 2

    #t = AveragePooling2D(4)(t)
    # t = Flatten()(t)
    outputs = Dense(y.shape[1], activation='softmax')(t)

    model = Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def kaggle_save(y_pred, eval_df):
    with open("gbr_predictions.txt", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["TweetID", "NoRetweets"])
        for index, prediction in enumerate(y_pred):
            writer.writerow([str(eval_df['id'].iloc[index]), str(int(prediction))])

if __name__ == '__main__':
    print("Getting data..")
    train_df, eval_df, train_labels, eval_labels= get_data()
    print("Done")
    
    print("Getting model..")
    model = create_res_net(train_df, train_labels)
    print("Done")
    model.summary()
    print("start compiling")
    model.compile(optimizer="Adam", loss="mse", metrics=["mae", "acc"])
    print("Done")
    print("Start fit")
    model.fit(train_df, train_labels)
    print("Done")
    print("Start eval")
    res = model.evaluate(eval_df)
    print("done")
    print(res)
    #print(res.shape)

