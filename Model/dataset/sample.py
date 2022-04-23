import json
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import psutil
import sys
gc.set_threshold(100000)
#mem=psutil.virtual_memory()
def read_json(filename):
    with open(filename, 'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict


def trp(l, n,INT=0):#将l长度后尾补0到长度n
    """ Truncate or pad a list """
    r = l[:n]

    if len(r) < n:
        r.extend(list([INT]) * (n - len(r)))


    return r


def down_sample(logs, labels, sample_ratio):#按比例选定样本
    print('sampling...')
    total_num = len(labels)#获取总标签数
    all_index = list(range(total_num))#获取下标
    sample_logs = {}
    for key in logs.keys():
        sample_logs[key] = []
    sample_labels = []
    sample_num = int(total_num * sample_ratio)

    for i in tqdm(range(sample_num)):
        random_index = int(np.random.uniform(0, len(all_index)))
        for key in logs.keys():
            sample_logs[key].append(logs[key][random_index])
        sample_labels.append(labels[random_index])
        del all_index[random_index]
    return sample_logs, sample_labels


def sliding_window(data_dir, datatype, window_size,Event_TF_IDF,template, sample_ratio=1,data_type='HDFS'):#滑动窗口，用于序列预测，窗口大小的序列，预测的下一个事件是什么
    '''
    dataset structure
        result_logs(dict):
            result_logs['feature0'] = list()
            result_logs['feature1'] = list()
            ...
        labels(list)
    '''

    if data_type == "HDFS":
        event2semantic_vec = read_json(data_dir + 'HDFS/events_semantic.json')  # 加载事件语义向量
    elif data_type == "BGL":
        event2semantic_vec = read_json(data_dir + 'BGL/events_semantic.json')
    num_sessions = 0
    result_logs = {}
    result_logs['Sequentials'] = []
    result_logs['Quantitatives'] = []
    result_logs['Semantics'] = []
    labels = []
    if data_type=='HDFS' :
      if datatype == 'train':
        data_dir += 'HDFS/hdfs_train'
      if datatype == 'val':
        data_dir += 'HDFS/hdfs_test_normal'
      if datatype == 'test_ab':
        data_dir += 'HDFS/hdfs_test_abnormal'
    elif data_type=='BGL' :
        if datatype == 'train':
            data_dir += 'BGL/bgl_train'
        if datatype == 'val':
            data_dir += 'BGL/bgl_test_normal'
        if datatype == 'test_ab':
            data_dir += 'BGL/bgl_test_abnormal'

    count_idf(template)  # 计算每个事件的IDF
    template=pd.read_csv(template)
    print("loading")
    with open(data_dir, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            if data_type=="HDFS":
                line = tuple(map(lambda n: n , map(int, line.strip().split())))#将训练文件的每行分为int列表，（并每个元素-1）去掉
            if data_type=="BGL":
                line = tuple(map(lambda n: n, map(int, line.strip().split())))  # 将训练文件的每行分为int列表，并每个元素
#考虑一行不足window_size情况
            #print("line",line)
            for i in range(len(line) - window_size):
                Sequential_pattern = list(line[i:i + window_size])#从第i个元素到第i+window_size个元素，但不包括第i+windowsize元素
                Quantitative_pattern = [0] * 2000
                log_counter = Counter(Sequential_pattern)

                for key in log_counter:
                    Quantitative_pattern[key] = log_counter[key]
                Semantic_pattern = []
                Event_TF=dict()
                for event in Sequential_pattern:
                    if  event == 0:  #
                        Semantic_pattern.append([0] * 300)
                    elif type(event2semantic_vec[str(event - 1)]) == type(0.1):
                        print("error,event:", Sequential_pattern)
                    else:

                        if Event_TF_IDF == 1:
                            if event in Event_TF.keys():
                                TF = Event_TF[event]
                                idf = template.loc[event - 1, "idf"]
                                # print("Event",event,"TF",TF)
                            else:
                                if len(Sequential_pattern) == 0:
                                    TF = 0
                                    idf = template.loc[event - 1, "idf"]
                                    Event_TF[event] = TF
                                else:
                                    TF = Sequential_pattern.count(event) / len(Sequential_pattern)
                                    Event_TF[event] = TF
                                    # print("index:",template.columns)
                                    idf = template.loc[event - 1, "idf"]
                                    # print("event",event,"tf:",TF,"idf:",idf)

                            #TF_IDF = TF * idf
                            Semantic_pattern.append(event2semantic_vec[str(event - 1)])



                        else:
                            Semantic_pattern.append(event2semantic_vec[str(event - 1)])
                #print("semantic:",Semantic_pattern)
                #print("seq:", Sequential_pattern,line[i + window_size])
                #print("sem:",Semantic_pattern)
                Sequential_pattern = np.array(Sequential_pattern)[:,
                                                                  np.newaxis]
                Quantitative_pattern = np.array(
                    Quantitative_pattern)[:, np.newaxis]
                result_logs['Sequentials'].append(Sequential_pattern)
                result_logs['Quantitatives'].append(Quantitative_pattern)
                result_logs['Semantics'].append(Semantic_pattern)

                labels.append(line[i + window_size])#将下标i+window_size的元素作为预测的标签对象，也就是窗口序列的下一个事件是什么
    print("labels:",len(set(labels)))
    if sample_ratio != 1:
        result_logs, labels = down_sample(result_logs, labels, sample_ratio)

    print('File {}, number of sessions {}'.format(data_dir, num_sessions))
    print('File {}, number of seqs {}'.format(data_dir,
                                              len(result_logs['Sequentials'])))

    return result_logs, labels

def count_idf(template_path):
    template=pd.read_csv(template_path)
    template["idf"]=""
    for i in range(len(template)):
        template.loc[i,"idf"]=np.log(float(template["Occurrences"].sum()) / (template.loc[i,"Occurrences"] + 1))
    template.to_csv(template_path,index=None)

def session_window(data_dir, datatype, Event_TF_IDF,template,sample_ratio=1,data_type='HDFS',):
    if data_type=="HDFS":
        event2semantic_vec = read_json(data_dir + 'HDFS/events_semantic.json') #加载事件语义向量
    elif data_type=="BGL":
        event2semantic_vec = read_json(data_dir + 'BGL/events_semantic.json')
    elif data_type=="OpenStack":
        event2semantic_vec=read_json(data_dir+'OpenStack/events_semantic.json')
    elif data_type=="Thunderbird":
        event2semantic_vec=read_json(data_dir+'Thunderbird/events_semantic.json')
    result_logs = {}
    result_logs['Sequentials'] = []
    result_logs['Quantitatives'] = []
    result_logs['Semantics'] = []
    labels = []
    if datatype=="train":
        data_dir+=(data_type+"/MLog_log_train.csv")
    if datatype=="val":
        data_dir+=(data_type+"/MLog_log_valid.csv")
    if datatype=="test":
        data_dir += (data_type + "/MLog_log_test.csv")


    #count_idf(template)#计算每个事件的IDF

    template=pd.read_csv(template)
    train_df = pd.read_csv(data_dir)
    #used_mem = float(mem.used)
    Event_TF = dict()
    for i in tqdm(range(len(train_df))):
        ori_seq = [
            int(eventid) for eventid in train_df["Sequence"][i].split(' ')#将sequence列中每行数据分割成int列表
        ]

        Sequential_pattern = trp(ori_seq, 50)#长度补0到50

        #print("Seq:",Sequential_pattern)
        temp=[0]*768
        Semantic_pattern = []#语义矩阵
        #print(Sequential_pattern)
        Event_TF.clear()#字典，记录已计算的时间的TF值
        for event in Sequential_pattern:
            #print("event:",event)

            if  event == 0: #事件序列的0表示为-1
                Semantic_pattern.append(temp)
            #elif type(event2semantic_vec[str(event - 1)])==type(0.1):
                #print("event:",Sequential_pattern)
            else:

                #print("EVENT",event-1)
                if Event_TF_IDF==1:
                    """
                    if event in Event_TF.keys():
                        #TF=Event_TF[event]
                        #idf = template.loc[event - 1, "idf"]
                        #print("Event",event,"TF",TF)
                    else:
                        #print("len:",len(Sequential_pattern))
                        if len(Sequential_pattern)==0:
                            TF=0
                            #idf = template.loc[event - 1, "idf"]
                            Event_TF[event]=TF
                        else:
                            TF=Sequential_pattern.count(event)/len(Sequential_pattern)
                            Event_TF[event]=TF
                            #print("index:",template.columns)
                            #idf = template.loc[event-1,"idf"]
                            #print("event",event,"tf:",TF,"idf:",idf)

                    #TF_IDF=TF*idf
                    #vec=map(lambda x:x*TF_IDF,event2semantic_vec[str(event - 1)])
                    #vec=np.array(event2semantic_vec[str(event - 1)],dtype='float32')

                    #vec=(vec*TF_IDF).astype(np.float32)
                    #print("vec:",vec)
                    #Semantic_pattern.append([i*TF_IDF for i in event2semantic_vec[str(event - 1)]])
                    #del vec
                    """
                    Semantic_pattern.append(event2semantic_vec[str(event - 1)])
                    #used_mem2 = float(mem.used)
                    #print("mem_used:", used_mem2 - used_mem)

                    #Semantic_pattern.append((np.array(event2semantic_vec[str(event - 1)])*TF_IDF).tolist())
                else:
                    Semantic_pattern.append(event2semantic_vec[str(event - 1)])

                #print("event2sem",event2semantic_vec[str(event - 1)])
        Quantitative_pattern = [0] * 2000 #数量特征,定义为dict字典，上限29个,HDFS定义50，BGL根据需要改
        log_counter = Counter(Sequential_pattern)#对每一行事件计数，按关键字访问

        for key in log_counter:
            Quantitative_pattern[key] = log_counter[key]

        Sequential_pattern = np.array(Sequential_pattern)[:, np.newaxis]#给序列特征增加一维
        Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]#给计数特征增加一维
        result_logs['Sequentials'].append(Sequential_pattern)
        result_logs['Quantitatives'].append(Quantitative_pattern)

        result_logs['Semantics'].append(Semantic_pattern)

        labels.append(int(train_df["label"][i]))
    #pd.DataFrame(result_logs['Semantics']).to_csv("Semantic.csv")
    if sample_ratio != 1:
        result_logs, labels = down_sample(result_logs, labels, sample_ratio)

    # result_logs, labels = up_sample(result_logs, labels)

    print('Number of sessions({}): {}'.format(data_dir,
                                              len(result_logs['Semantics'])))
    #返回日志数和标签
    return result_logs, labels
