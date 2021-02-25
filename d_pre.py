#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from profanity_check import predict, predict_prob
from nltk import FreqDist

import string
import re



def tokenize(df):
    c_df = df.copy()
    col_name = 'comment_text'
    
    # 토크나이저로 전체 문장을 단어별로 분할, 토크나이저는 RegexTokenizer사용
    c_df[col_name] = c_df[col_name].apply(lambda x : RegexpTokenizer(r'[a-z]+').tokenize(x.lower()))
    
    # Lemmatizer로 단어를 원형으로 표기
    l = WordNetLemmatizer()
    c_df[col_name] = c_df[col_name].apply(lambda x : [l.lemmatize(word) for word in x])
    
    # StopWords 제거
    stop_words = stopwords.words('english')
    c_df[col_name] = c_df[col_name].apply(lambda x : [word for word in x if word not in stop_words])
        
    
    return c_df



def make_features(df):
    
    c_df = df.copy()
    
    col_name = 'comment_text'
    
    
    # 전체 문장의 길이
    c_df['tot_len'] = c_df[col_name].apply(lambda x : len(x))
       
    
    # 개행문자의 개수
    c_df['nl'] = c_df[col_name].str.findall(r'\n').str.len()
    
    
    # you 라는 지칭대명사의 count개수
    c_df['you'] = c_df[col_name].str.lower().str.findall(r'you').str.len()
    
    
    # 대문자가 얼마나 포함되어있는가
    c_df['Cap'] = c_df[col_name].str.findall(r'[A-Z]').str.len()
    
    
    # 느낌표 사용횟수
    c_df['exclamation'] = c_df[col_name].str.findall(r'\!').str.len()
    
    
    # 물음표 사용횟수
    c_df['question'] = c_df[col_name].str.findall(r'\?').str.len()
    
    
    # smile_emoji 사용여부
    smile_emoji = [':-)', ':)', ';-)', ';)', ':P', ';P', ':D', ';D', '<3']
    c_df['smile'] = c_df[col_name].apply(lambda x : sum([1 for word in x.split() if word in smile_emoji]))
    
    
    # Stopword 개수 파악
    stop_words = set(stopwords.words('english'))
    c_df['stop_words'] = c_df[col_name].apply(lambda x: sum([1 for word in x.split() if word in stop_words]))
    
    
    # Punctuation 사용 개수 파악
    c_df['punc'] = c_df[col_name].apply(lambda x: sum([1 for word in x.split() if word in string.punctuation]))
    
    
    return c_df.drop(columns=col_name, axis=1)




# make_features2 함수의 경우, 인자 값으로 t_pre를 수행한 df를 받는다.
def make_features2(df):
    c_df = df.copy()
    
    col_name = 'comment_text'
    
    # 비속어 포함개수
    c_df['profanity'] = c_df[col_name].apply(lambda x : predict(x).sum() if len(x) > 0 else 0)
    
    # 문장 내에서 반복되는 단어의 최빈값
    c_df['most_rep'] = c_df[col_name].apply(lambda x : FreqDist(np.hstack(x)).most_common(1)[0][1] if len(x) > 0 else 0)
    
    return c_df.drop(columns=col_name, axis=1)

