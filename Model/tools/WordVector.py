# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:21:23 2020

@author: 梁坤
"""
import torch
import pandas as pd
import numpy as np
import bcolz
import pickle
import json
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
from Model.tools.Bert2vector import *
from sklearn import preprocessing
struct_log = '../data/hdfs/HDFS_100k.log_structured.csv'#结构化日志文件
vector_file = '../../data/glove.840B.300d.txt'#glove文件
# vector_file='../data/test.txt'
vector_size = 300  # 下载的glove文件的维度

def normalization(data):
    _range = np.max(data) - np.min(data)
    if np.max(data)==np.min(data):
        return (data - np.min(data)+0.5) / (_range+1)
    return (data - np.min(data)) / _range

def detect_events(dir=struct_log):  # 根据日志解析生成的初步结构化文件，提取所有事件
    if dir.endswith('.csv'):
        print("Loading", dir)
        struct_log = pd.read_csv(dir, engine='c', na_filter=False, memory_map=True)
        data_dict = dict()  # 字典
        for idx, row in struct_log.iterrows():

            if (int(row["EventId"].replace('E', '')) not in data_dict):
                data_dict[int(row["EventId"].replace('E', ''))] = row["EventTemplate"]

        # for key in data_dict:
        # print(key,":",data_dict[key],"\n")
        for i in sorted(data_dict):
            print(i, ":", data_dict[i], "\n")
    return data_dict

def detect_Events(Events_mapping,dir=struct_log):  # 根据日志解析生成的初步结构化文件，提取所有事件模板
        if dir.endswith('.csv'):
            print("Loading", dir)
            struct_log = pd.read_csv(dir, engine='c', na_filter=False, memory_map=True)
            print("load over")
            data_dict = dict()  # 字典
            for idx, row in struct_log.iterrows():
              if row["EventId"] in Events_mapping:
                id=Events_mapping.index(row["EventId"])
                if id not in data_dict:
                    data_dict[id]= row["EventTemplate"]

            # for key in data_dict:
            # print(key,":",data_dict[key],"\n")
            for i in sorted(data_dict):
                print(i, ":", data_dict[i], "\n")
        return data_dict
    # 根据预训练文件，事件列表，嵌入词向量表
def Detect_Events(dir='../data/HDFS/HDFS.log_templates.csv'): #根据模板文件template直接提取模板
    if dir.endswith('.csv'):
        print("Loading", dir)
        Events_list= pd.read_csv(dir, engine='c', na_filter=False, memory_map=True)
        print("load over")
        data_dict = dict()  # 字典
        Events=Events_list["EventId"].tolist()
        Events.insert(0, "#")
        Events_template=Events_list["EventTemplate"].tolist()
        Events_template.insert(0, "#")
        for i,template in enumerate(Events_template,start=0):
            if i==0: continue
            data_dict[i]=template
        # for key in data_dict:
        # print(key,":",data_dict[key],"\n")
        for i in sorted(data_dict):
            print(i, ":", data_dict[i], "\n")
    return data_dict


def generate(data_dict, vocab_size=10000, embed_size=300):  # 1w个单词，300维度向量
    return


def save_words_info(vector_file):  # 根据下载的预训练glove词向量，保存词向量信息,方便后续查表
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir=f'vectors.dat', mode='w')
    print(len(vectors))
    num_file = sum([1 for i in open(vector_file, "rb")])
    # 以二进制方式读取
    with open(vector_file, 'rb') as f:
        for l in tqdm(f, total=num_file):
            # i=i+1
            # decode还原二进制编码
            line = l.decode().strip().split(' ')
            # vect = np.array(line[1:]).astype(np.float)
            vect = np.array(line[1:]).astype(np.float)
            if (len(vect) != vector_size):  # 发现某个单词的向量维度不准确，打印出来，跳过这个词继续统计
                print(len(vect), ":", line)
                continue
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            # print(line, line[1:])
            vectors.append(vect)

    print("len", len(vectors))
    print("words_num", idx)
    # 根据统计出的glove表中单词数目和维度，将统计的vectors从第一行（第0行为初始1）开始化为维度300的二维向量，保存
    vectors = bcolz.carray(vectors[1:].reshape(idx, vector_size), rootdir=f'vectors.dat', mode='w')
    vectors.flush()
    # 保存glove中的单词表
    pickle.dump(words, open(f'words.pkl', 'wb'))
    # 保存glove中单词对应下标的关系，下标用于寻找向量
    pickle.dump(word2idx, open(f'idx.pkl', 'wb'))


# embed
# embeds.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
def Word2Vector(word):  # 将一个单词化为词向量

    vectors = bcolz.open(f'vectors.dat')[:]
    words = pickle.load(open(f'words.pkl', 'rb'))
    word2idx = pickle.load(open(f'idx.pkl', 'rb'))
    glove = {word: vectors[word2idx[word]]}
    return glove[word]

"""vectors = bcolz.open(f'vectors.dat')[:]
words = pickle.load(open(f'words.pkl', 'rb'))
word2idx = pickle.load(open(f'idx.pkl', 'rb'))"""
vectors = bcolz.open(f'../Model/tools/vectors.dat')[:]
words = pickle.load(open(f'../Model/tools/words.pkl', 'rb'))
word2idx = pickle.load(open(f'../Model/tools/idx.pkl', 'rb'))
def Sentenci2Vector(sentence,tf_idf=1,template='../data/HDFS/HDFS.log_templates.csv'):  # 将一句话化成二维的词向量组，行表示每个单词，列表示单词的维度
    """vectors = bcolz.open(f'../Model/tools/vectors.dat')[:]
    words = pickle.load(open(f'../Model/tools/words.pkl', 'rb'))
    word2idx = pickle.load(open(f'../Model/tools/idx.pkl', 'rb'))"""
    sentence_vector = []
    sentence = pre_handle(sentence)  # 预处理，删去符号和数字，提取词干
    sentence = sentence.strip().split()
    # print(sentence)
    tf_idf_vect=[]
    totallogs=0
    templates = pd.read_csv(template, engine='c')
    for index,row in templates.iterrows():
        totallogs+=int(row['Occurrences'])
    print("total logs:", totallogs)
    print(sentence)
    for i, word in enumerate(sentence):
        #print(sentence[i])
        if sentence[i] not in word2idx:
            sentence_vector.append([0.0]*300)

        #print(sentence[i])
        else:
            vect = vectors[word2idx[sentence[i]]]
        # print(vect)
            sentence_vector.append(vect)
        if tf_idf==1:
            tf=float(sentence.count(word))/len(sentence)
            idf0=0
            for index,row in templates.iterrows():
                if word in row["EventTemplate"]:
                    #print("Event:",row["EventTemplate"])
                    idf0+=row['Occurrences']
            idf=np.log(float(totallogs)/(idf0+1))
            print("idf:",idf)
            tf_idf_vect.append(tf*idf)
            print(word,":",tf_idf_vect[-1])

    print("tf_idf_vect:",tf_idf_vect)
    if tf_idf==1:
        if len(tf_idf_vect)!=0:
            tf_idf_vect=normalization(tf_idf_vect)
    print("tf_idf_vect:", tf_idf_vect)
    if tf_idf==1:
        sentence_vector=np.array(sentence_vector)
        print("sentence_len:",len(sentence_vector))
        for i in range(len(sentence_vector)):
            #print(sentence_vector[i])
            sentence_vector[i]=tf_idf_vect[i]*sentence_vector[i]
            #print(sentence_vector[i])
    #print("tf_idf_vect",tf_idf_vect)
    #print("sentence_vector",  sentence_vector.shape)
    if len(sentence_vector)==0:
        sentence_vector=sentence_vector.tolist().append([0.0]*300)
    return sentence_vector

def MutiSentence2Vector(x_in):
    vectors = bcolz.open(f'../tools/vectors.dat')[:]
    words = pickle.load(open(f'../tools/words.pkl', 'rb'))
    word2idx = pickle.load(open(f'../tools/idx.pkl', 'rb'))

    for index,str in enumerate(x_in):
        sentence=str.strip().split()
        sentence_vector = []
        for i, word in enumerate(sentence):
          #print(sentence[i])
          if sentence[i] not in word2idx:
              continue
          #print(sentence[i])
          vect = vectors[word2idx[sentence[i]]]
          #print("vect",vect.shape)
          # print(vect)
          sentence_vector.append(vect.tolist())

        tensor = torch.tensor(sentence_vector)
        means = torch.mean(tensor, dim=0).numpy()#求平均
        #print("means_shape",means.shape)
        try:
            means.resize(300)
            #print("means_shape", means.shape)
        except Exception:

            means = np.zeros((300), dtype=np.float)
            print("Error:",index," ",str)
        if index==0:
            x_out=means
        else:
            x_out=np.vstack((x_out,means))
    #print("x_out_length",x_out)

    #print("x_out",x_out)
    x_out=pd.DataFrame(x_out,columns=[x for x in range(300)])
    print("x_shape:",x_out.shape)
    return x_out


    return sentence_vector

def MutiSentence2Vector2(x_in):
    vectors = bcolz.open(f'../tools/vectors.dat')[:]
    words = pickle.load(open(f'../tools/words.pkl', 'rb'))
    word2idx = pickle.load(open(f'../tools/idx.pkl', 'rb'))

    for index,str in enumerate(x_in):
        sentence=str.strip().split()
        sentence_vector = []
        for i, word in enumerate(sentence):
          #print(sentence[i])
          if sentence[i] not in word2idx:
              continue
          #print(sentence[i])
          vect = vectors[word2idx[sentence[i]]]
          #print("vect",vect.shape)
          # print(vect)
          sentence_vector.append(vect.tolist())
        if(len(sentence_vector)==0):
            str_list=list(str)
            sstr = ' '.join(str_list)
            sentence = sstr.strip().split()
            for i, word in enumerate(sentence):
                # print(sentence[i])
                if sentence[i] not in word2idx:
                    continue
                # print(sentence[i])
                vect = vectors[word2idx[sentence[i]]]
                # print("vect",vect.shape)
                # print(vect)
                sentence_vector.append(vect.tolist())
        tensor = torch.tensor(sentence_vector)
        means = torch.mean(tensor, dim=0).numpy()#求平均
        #print("means_shape",means.shape)
        try:
            means.resize(300)
            #print("means_shape", means.shape)
        except Exception:

            means = np.zeros((300), dtype=np.float)
            print("Error:",index," ",str)
        if index==0:#结果拼接到x_out中
            x_out=means
        else:
            x_out=np.vstack((x_out,means))
    #print("x_out_length",x_out)

    #print("x_out",x_out)
    x_out=pd.DataFrame(x_out,columns=[x for x in range(300)])
    print("x_shape:",x_out.shape)
    return x_out


    return sentence_vector

def add_sentence_vector(sentence,tf_idf=1,template=None):  # 将一句话化成单行词向量，即每个单词向量求平均
    if tf_idf==1:
        sentence_vector = Sentenci2Vector(sentence,tf_idf=1,template=template)
    else:
        sentence_vector = Sentenci2Vector(sentence, tf_idf=0,template=template)
    tensor = torch.tensor(sentence_vector)
    #print(tensor)
    out = torch.mean(tensor, dim=0)
    out_array = out.numpy().tolist()  # 数组需要先转为列表，然后装入dict，便于转换成json

    return out_array


def pre_handle(sentence):  # 对句子进行预处理，提取词干,去掉符号和数字,
    str_list = list(sentence)  # 字符串转化为列表，单个字符
    # print(str_list)
    for i, char in enumerate(str_list):  # 只保留字母，别的符号和数字用空格代替
        if char >= 'a' and char <= 'z':
            continue
        elif char >= 'A' and char <= 'Z':
            if i > 0 and str_list[i - 1] >= 'a' and str_list[i - 1] <= 'z':
                str_list.insert(i, ' ')
            if i > 0 and str_list[i - 1] >= 'A' and str_list[i - 1] <= 'Z' and i+1<len(str_list) and str_list[i + 1] >= 'a' and str_list[i + 1] <= 'z':
                str_list.insert(i, ' ')
        elif char>='0' and char<='9':
            if i > 0 and str_list[i - 1] >= 'a' and str_list[i - 1] <= 'z':
                str_list.insert(i, ' ')
            if i+1<len(str_list) and str_list[i + 1] >= 'a' and str_list[i + 1] <= 'z':
                str_list.insert(i+1, ' ')
            continue
        elif char==' ':
            continue
        else:
            str_list[i]=' '

    # 将列表组合成字符串
    sstr = ''.join(str_list)
    return sstr




def tf_idf_event_vector(total_files="../../data/hdfs/MLog_log_train.csv",template='../../data/HDFS/HDFS.log_templates.csv',aim='../data/HDFS/tf-idf-vector.csv'):
    total=pd.read_csv(total_files, engine='c', na_filter=False, memory_map=True)
    totalLogs=[]
    for index,row in total.iterrows():
        #print(row['Sequence'])
        totalLogs.extend(row["Sequence"].split())
    totalNum=len(totalLogs)
    print("total logs:",totalNum)

    templates=pd.read_csv(template,engine='c')
    Events_tfidf=[]
    word_num=0
    for index,row in templates.iterrows():

        for word in row["EventTemplate"].replace("").split():
            print(word,":")





    return


def Events_vectors(events_dict,events_semantic="../data/hdfs/events_semantic.json",tf_idf=1,template=None):
    out_dict = dict()
    for key in sorted(events_dict):
        if tf_idf==1:
            events_dict[key] = add_sentence_vector(events_dict[key],tf_idf=1,template=template)
        else:
            events_dict[key] = add_sentence_vector(events_dict[key],tf_idf=0,template=template)
        #events_dict[key] = Bert2vector(list(events_dict[key]))
        out_dict[key - 1] = events_dict[key]
    #print("out_dict",out_dict)
    #events_json = json.dumps(out_dict)
    json_str = json.dumps(out_dict, indent=4)
    with open(events_semantic, 'w') as json_file:
        json_file.write(json_str)

    return json_str

def count_idf(template_path):
    template=pd.read_csv(template_path)
    template["idf"]=""
    for i in range(len(template)):
        template.loc[i,"idf"]=np.log(float(template["Occurrences"].sum()) / (template.loc[i,"Occurrences"] + 1))
    template.to_csv(template_path,index=None)

def count_idf(template_path,scalermin,scalermax):
    template=pd.read_csv(template_path)
    template["idf"]=""
    minmax=preprocessing.MinMaxScaler(feature_range=(scalermin,scalermax),copy=False)
    for i in range(len(template)):
        template.loc[i,"idf"]=np.log(float(template["Occurrences"].sum()) / (template.loc[i,"Occurrences"] + 1))
    template["idf"]=minmax.fit_transform(np.array(template["idf"]).reshape(-1,1))
    print("template:",template["idf"])
    template.to_csv(template_path,index=None)

def Events_Bert_vectors(template,tf_idf=1,IDF=1,scalermin=1,scalermax=2,events_semantic="../data/HDFS/events_semantic.json"):
    temp=pd.read_csv(template)
    Events=list(temp["EventTemplate"])
    #Events.insert(0, "#")
    print("transfering templates 2 vector···")

    out_dict=dict()
    count_idf(template,scalermin,scalermax)  # 计算每个事件的IDF
    for i,Event in tqdm(enumerate(Events)):
        #events_dict[key] = add_sentence_vector(events_dict[key])
        if IDF==0:
            out_dict[i] = word2vector3(Event,template,tf_idf)
        elif IDF==1:

                out_dict[i]=(np.array(word2vector3(Event,template,tf_idf))*temp.at[i,"idf"]).tolist()

        elif IDF==2:
            vect=word2vector3(Event,template,tf_idf)
            vect.append(temp.at[i,"idf"])
            out_dict[i]=vect
            #print(out_dict[i])
        #print(key-1," ")
        #print("out_dict",events_dict[key])
    #print("out_dict",out_dict)
    #events_json = json.dumps(out_dict)
    json_str = json.dumps(out_dict, indent=4)
    with open(events_semantic, 'w') as json_file:
        json_file.write(json_str)

    return json_str


if __name__=='__main__':
    # print(pre_handle("Exception in receiveBlock for block <*> <*> "))

    # Events_dict = detect_events()
     #Events_vectors(Events_dict)

    #save_words_info(vector_file)

    Events_Bert_vectors(template='../data/HDFS/HDFS.log_template.csv')

    #tf_idf_event_vector()
