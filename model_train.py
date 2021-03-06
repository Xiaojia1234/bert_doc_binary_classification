# -*- coding: utf-8 -*-
# author: Jclian91
# place: Pudong Shanghai
# time: 2020-02-12 13:37

'''
使用/修改/注释：@Amy
    第二个程序模块
    数据预处理之后，利用Bert提取文档的特征，每个文档的填充长度为100，对应一个768维的向量；
    然后用Keras创建DNN来进行模型训练；
    然后对测试集进行验证，并保存模型文件，便于后续的模型预测使用。

'''


import os
# 使用GPU训练
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7,8"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from load_data import train_df, test_df
from keras.utils import to_categorical
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, BatchNormalization, Dense
from bert.extract_feature import BertVector



# 读取文件并进行转换
bert_model = BertVector(pooling_strategy="REDUCE_MEAN", max_seq_len=100)
print('begin encoding')
f = lambda text: bert_model.encode([text])["encodes"][0]
train_df['x'] = train_df['text'].apply(f)
print('train_df[\'x\']:',len(train_df['x']),'train_df[\'x\'][0].shape',train_df['x'][0].shape,'\n',train_df['x'][:5])
test_df['x'] = test_df['text'].apply(f)
print('test_df[\'x\']:',len(test_df['x']),'test_df[\'x\'][0].shape()',test_df['x'][0].shape,'\n',test_df['x'][:5])
print('end encoding')
#
x_train = np.array([vec for vec in train_df['x']])
x_test = np.array([vec for vec in test_df['x']])
y_train = np.array([vec for vec in train_df['label']])
y_test = np.array([vec for vec in test_df['label']])
print('x_train: ', x_train.shape)
print('x_test: ', x_test.shape)

# Convert class vectors to binary class matrices.
#
num_classes = 2
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
print('y_train : ', y_train .shape)
print('y_test: ', y_test.shape)

#
# 创建模型
x_in = Input(shape=(768, ))
x_out = Dense(32, activation="relu")(x_in)
x_out = BatchNormalization()(x_out)
x_out = Dense(num_classes, activation="softmax")(x_out)
model = Model(inputs=x_in, outputs=x_out)
print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
#
# 模型训练以及评估
model.fit(x_train, y_train, batch_size=8, epochs=20)
model.save('visit_classify.h5')
# print(model.evaluate(x_test, y_test))