# -*- coding: utf-8 -*-
# author: Jclian91
# place: Pudong Shanghai
# time: 2020-02-12 12:57
import pandas as pd


# 读取txt文件
def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = [_.strip() for _ in f.readlines()]
    # print('content[0]:',len(content[0]),content[0])
    labels, texts = [], []
    for line in content:
        parts = line.split()
        # print('parts:',len(parts),parts)
        label, text = parts[0], ''.join(parts[1:])
        labels.append(label)
        texts.append(text)

    return labels, texts


file_path = 'data/train.txt'
labels, texts = read_txt_file(file_path)
# print('labels:',len(labels),labels)
# print('texts:',len(texts),texts)
train_df = pd.DataFrame({'label': labels, 'text': texts})

file_path = 'data/test.txt'
labels, texts = read_txt_file(file_path)
test_df = pd.DataFrame({'label': labels, 'text': texts})

# print(train_df.head())
# print(test_df.head())

train_df['text_len'] = train_df['text'].apply(lambda x: len(x))
print(train_df.describe())
#从这个结构展示可以看出，训练数据的文本长度的75%分位点为100.75，所以后面模型训练时的padding长度统一选取100
# print('train_df[\'text_len\']:',train_df['text_len'])
