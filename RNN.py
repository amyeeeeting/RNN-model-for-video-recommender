import pandas as pd
import numpy as np
import sys
import math
import argparse
import csv
import json
from tqdm import tqdm
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.layers import Input, Dense, Embedding, concatenate, Flatten, Activation, Add, Dropout, Multiply, LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import os

### data preprocessing ###
# user id : list of [item_id]
user_item_train={}
counter=0
with open('input/train_eventid_filtered.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        user_item_train[row[0]]=row[1:]

user_item_test={}
counter=0
with open('input/test_eventid_filtered.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        user_item_test[row[0]]=row[1:]

with open('input/eventid2token.json','r') as f:
    eventid2token=json.load(f)
    
user_token_train = {}
for user_id, item_ids in user_item_train.items():
    user_token_train[user_id] = list(set([ eventid2token[item_id] for item_id in item_ids if item_id in eventid2token]))
    
user_token_test = {}
for user_id, item_ids in user_item_test.items():
    user_token_test[user_id] = list(set([ eventid2token[item_id] for item_id in item_ids if item_id in eventid2token]))

num_users=len(set(user_token_train.keys())|set(user_token_test.keys()))
num_items=len(set(np.hstack(user_token_train.values())) | set(np.hstack(user_token_test.values())))

user2idx = {user:idx for idx, user in enumerate(list(set(user_token_train.keys())|set(user_token_test.keys())))}
item2idx = {item:idx for idx, item in enumerate(list(set(np.hstack(user_token_train.values())) | set(np.hstack(user_token_test.values()))))} 

#sequence_lenth = max_len
#train_item_sequences(flatten user_id watched item_id, max_len+1(label))
#train_user 1D list : all items
max_len = 20
train_item_sequences = []
train_user = []
for user_id, items in tqdm(user_token_train.items()):
    train_item_list = items
    train_item_list = [item2idx[item_id] for item_id in train_item_list]
    train_item_list = [0]*max_len + train_item_list
    for i in range(len(items)):
        train_item_sequences.append(train_item_list[i:i+max_len+1])
    train_user.append([user2idx[user_id]]*len(items))

test_item_sequences = []
test_user = []
for user_id, items in tqdm(user_token_test.items()):
    test_item_list = items
    test_item_list = [item2idx[item_id] for item_id in test_item_list]
    test_item_list = [0]*max_len + test_item_list
    for i in range(len(items)):
        test_item_sequences.append(test_item_list[i:i+max_len+1])
    test_user.append([user2idx[user_id]]*len(items))

train_item_seq = []
train_labels = []
for seq in train_item_sequences:
    train_item_seq.append(seq[:max_len])
    train_labels.append(seq[-1])
test_item_seq = []
test_labels = []
for seq in test_item_sequences:
    test_item_seq.append(seq[:max_len])
    test_labels.append(seq[-1])

train_user_ids = np.hstack(train_user)
train_item_seq = np.array(train_item_seq)
train_labels = np.array(train_labels)
test_user_ids = np.hstack(test_user)
test_item_seq = np.array(test_item_seq)
test_labels = np.array(test_labels)

import gc
del eventid2token, user_token_train, train_item_sequences, train_user
gc.collect()

### build RNN model ###
def get_rnn_model():
    user_inp = Input((1,))
    user_hidden = Embedding(input_dim=num_users, output_dim=64)(user_inp)
    user_hidden = Flatten()(user_hidden)
    
    item_inp = Input((max_len,))
    item_hidden = Embedding(input_dim=num_items, output_dim=64)(item_inp)
    item_hidden = LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(item_hidden)
    
    hidden = concatenate([user_hidden, item_hidden])
    #hidden = LSTM(128, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(hidden)
    hidden = Dense(128, activation='relu')(hidden)
    hidden = Dropout(0.2)(hidden)
    
    output = Dense(num_items, activation='softmax')(hidden)
    
    model = Model(inputs=[user_inp, item_inp], outputs=output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model
model = get_rnn_model()
model.summary()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# restrict gpu usage
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

model.fit([train_user_ids, train_item_seq], train_labels, validation_split=0.1, batch_size=512, epochs=5)

### evaluation ###
item_rec = []
hits=[]
for user_id, seqs, target_item_id in tqdm(zip(test_user_ids, test_item_seq, test_labels)):
    predictions = model.predict([np.array(user_id).reshape(1,-1),seqs.reshape(1,-1)], batch_size=1024)
    ranks = np.flip(np.argsort(predictions), axis=0)
    hits.append(ranks == target_item_id)