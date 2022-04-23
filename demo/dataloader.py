"""
The interface to load log datasets. The datasets currently supported include
HDFS and BGL.

Authors:
    LogPAI Team

"""

import pandas as pd
import os
import numpy as np
import re
from sklearn.utils import shuffle
from collections import OrderedDict
from sklearn.feature_extraction.text import CountVectorizer
from Model.tools.WordVector import *
from ast import literal_eval
def func(x):
    if len(x)<1:
        return
    else: return list(map(int, x.split()))

def extract(train_file,train_abnormal_file=None,test_file=None,test_abnormal_file=None,need_abnormal=0,abnormal_num=500):



    if train_file.endswith('.csv'):
        train_df=pd.read_csv(train_file,engine="c")
        test_df=pd.read_csv(test_file,engine="c")
        x_train=[]
        #x_train=np.array(train_df["Sequence"].map(lambda x:list(map(int,x.split()))))#把sequence分割成数字放回sequence，再列表化后赋值给x_train
        for i,row in enumerate(train_df["Sequence"]):
            print("row:",row)
            x_train.append(list(map(int,row.split())))
        x_train=np.array(x_train)
        y_train = np.array(train_df["label"].map(lambda x: int(x)).tolist())
        print(test_df["Sequence"])
        #x_test= np.array(test_df["Sequence"].map(lambda x: list(map(int, x.split()))))  # 把sequence分割成数字放回sequence，再列表化后赋值给x_train
        x_test=[]
        for i,row in enumerate(test_df["Sequence"]):
            print("row",i,":",row)
            x_test.append(list(map(int,row.split())))
        x_test=np.array(x_test)
        y_test = np.array(test_df["label"].map(lambda x: int(x)).tolist())
        print("x_train:",x_train.shape)
        print("y_train:", y_train.shape)
        print("x_test:", x_test.shape)
        print("y_test:", y_test.shape)
        return (x_train, y_train), (x_test, y_test)
    else:
        x_train=[]
        y_train=None
        x_test=[]
        y_test=[]

        with open(train_file,"r") as train_f:
            with open(test_file,"r") as test_f:
                with open(test_abnormal_file,"r") as test_abnormal_f:
                    for line in train_f.readlines():
                        x_train.append(np.array(list(map(int,line.split()))))

                    for line in test_f.readlines():
                        x_test.append(np.array(list(map(int,line.split()))))
                        y_test.append(0)
                    for line in test_abnormal_f.readlines():
                        x_test.append(np.array(list(map(int,line.split()))))
                        y_test.append(1)
        x_train=np.array(x_train)

        if train_abnormal_file!=None:
            y_train=[]
            with open(train_abnormal_file,"r") as wf:
                for line in wf.readlines():
                    y_train.append(np.array(list(map(int,line.split()))))

        x_test=np.array(x_test)
        y_test=np.array(y_test)
        if need_abnormal==1:
            x_train=np.concatenate((x_train,x_test[y_test>0][:abnormal_num]),axis=0)
        print("x_train:", x_train.shape)
        #print("y_train:", y_train)
        print("x_test:", x_test.shape)
        print("y_test:", y_test.shape)
        return (x_train, y_train), (x_test, y_test)



def _split_data(x_data, y_data=None, train_ratio=0, split_type='uniform'):
    if split_type == 'uniform' and y_data is not None:
        pos_idx = y_data > 0
        print(pos_idx,end="\n")
        x_pos = x_data[pos_idx] #取出label为正的，x_pos
        print('x_pos')
        print(x_pos[0])
        y_pos = y_data[pos_idx]#取出label为正的，y_pos
        print(~pos_idx,end="\n")
        x_neg = x_data[~pos_idx]#取出label为负的（0）,x_neg
        print('x_neg')
        print(x_neg[0])
        y_neg = y_data[~pos_idx]#取出label为负的（0）,y_neg
        train_pos = int(train_ratio * x_pos.shape[0])
        train_neg = int(train_ratio * x_neg.shape[0])
       # print("test")
        #print(np.hstack([x_pos[0:1], x_neg[0:1]]))
        x_train = np.hstack([x_pos[0:train_pos], x_neg[0:train_neg]])#水平方向叠加
        y_train = np.hstack([y_pos[0:train_pos], y_neg[0:train_neg]])
        x_test = np.hstack([x_pos[train_pos:], x_neg[train_neg:]])
        y_test = np.hstack([y_pos[train_pos:], y_neg[train_neg:]])
        print("train_pos,train_neg")
        print(train_pos,end="\n")
        print(train_neg)
        print("x_train")
        print(x_train.shape[0])
    elif split_type == 'sequential':
        num_train = int(train_ratio * x_data.shape[0])
        x_train = x_data[0:num_train]
        x_test = x_data[num_train:]
        if y_data is None:
            y_train = None
            y_test = None
        else:
            y_train = y_data[0:num_train]
            y_test = y_data[num_train:]
    # Random shuffle
    indexes = shuffle(np.arange(x_train.shape[0]))
    x_train = x_train[indexes]#x,y特征重新随机排列
    if y_train is not None:
        y_train = y_train[indexes]
        print("x_test:")
        print(x_test)
    return (x_train, y_train), (x_test, y_test)

def load_HDFS(log_file, label_file=None, window='session', parameter_feature=1,train_ratio=0.7, split_type='sequential', save_csv=True, window_size=0):
    """ Load HDFS structured log into train and test data

    Arguments
    ---------
        log_file: str, the file path of structured log.
        label_file: str, the file path of anomaly labels, None for unlabeled data
        window: str, the window options including `session` (default).
        train_ratio: float, the ratio of training data for train/test split.
        split_type: `uniform` or `sequential`, which determines how to split dataset. `uniform` means
            to split positive samples and negative samples equally when setting label_file. `sequential`
            means to split the data sequentially without label_file. That is, the first part is for training,
            while the second part is for testing.

    Returns
    -------
        (x_train, y_train): the training data
        (x_test, y_test): the testing data
    """

    print('====== Input data summary ======')

    if log_file.endswith('.npz'):
        # Split training and validation set in a class-uniform way
        data = np.load(log_file,allow_pickle=True)
        x_data = data['x_data']
        print('x_data:\n',x_data)
        y_data = data['y_data']
       
        (x_train, y_train), (x_test, y_test) = _split_data(x_data, y_data, train_ratio, split_type)

    elif log_file.endswith('.csv'):
        assert window == 'session', "Only window=session is supported for HDFS dataset."
        print("Loading", log_file)
        struct_log = pd.read_csv(log_file, engine='c',
                na_filter=False, memory_map=True,error_bad_lines=False)
        print("load over.start handling to Event Sequence.")
        data_dict = OrderedDict()#有序hashmap字典-
        for idx, row in struct_log.iterrows():  #行列循环  按行迭代，返还行的索引以及行本身的内容
            blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])#匹配blk_或blk_-，数字1次或以上"
            blkId_set = set(blkId_list)#构建无序无重复blk集合"
            for blk_Id in blkId_set:          #为每个blkId匹配EventID，可能一个blk有多个EventId"
                if not blk_Id in data_dict:  #如果data_dict中没有这个blk，加上"
                    data_dict[blk_Id] = []
                data_dict[blk_Id].append(row['EventId'])#统计每个blk的事件
        data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])#将data_dict用items转化为数组后 矩阵化"
        print("data_df:")
        print(data_df)
        
        if label_file:
            # Split training and validation set in a class-uniform way
            label_data = pd.read_csv(label_file, engine='c', na_filter=False, memory_map=True)
            label_data = label_data.set_index('BlockId')#BlockId设置为索引"
            label_dict = label_data['Label'].to_dict()
            data_df['Label'] = data_df['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)#根据BlockID查找label字典，给矩阵添加label列"

            # Split train and test data
            #print(data_df['EventSequence'].values)
            (x_train, y_train), (x_test, y_test) = _split_data(data_df['EventSequence'].values, 
                data_df['Label'].values, train_ratio, split_type)
            print("x_train",x_train)

            print(y_train.sum(), y_test.sum())

        if save_csv:
            data_df.to_csv('../data/HDFS/data_instances.csv', index=False)
            #X_train_fit()
        if window_size > 0:
            x_train, window_y_train, y_train = slice_hdfs(x_train, y_train, window_size)
            x_test, window_y_test, y_test = slice_hdfs(x_test, y_test, window_size)
            log = "{} {} windows ({}/{} anomaly), {}/{} normal"
            print(log.format("Train:", x_train.shape[0], y_train.sum(), y_train.shape[0], (1-y_train).sum(), y_train.shape[0]))
            print(log.format("Test:", x_test.shape[0], y_test.sum(), y_test.shape[0], (1-y_test).sum(), y_test.shape[0]))
            return (x_train, window_y_train, y_train), (x_test, window_y_test, y_test)

        if label_file is None:
            if split_type == 'uniform':
                split_type = 'sequential'
                print('Warning: Only split_type=sequential is supported \
                if label_file=None.'.format(split_type))
            # Split training and validation set sequentially
            x_data = data_df['EventSequence'].values
            (x_train, _), (x_test, _) = _split_data(x_data, train_ratio=train_ratio, split_type=split_type)
            print('Total: {} instances, train: {} instances, test: {} instances'.format(
                  x_data.shape[0], x_train.shape[0], x_test.shape[0]))
            return (x_train, None), (x_test, None), data_df
    else:
        raise NotImplementedError('load_HDFS() only support csv and npz files!')

    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    num_total = num_train + num_test
    num_train_pos = sum(y_train)
    num_test_pos = sum(y_test)
    num_pos = num_train_pos + num_test_pos

    print('Total: {} instances, {} anomaly, {} normal' \
          .format(num_total, num_pos, num_total - num_pos))
    print('Train: {} instances, {} anomaly, {} normal' \
          .format(num_train, num_train_pos, num_train - num_train_pos))
    print('Test: {} instances, {} anomaly, {} normal\n' \
          .format(num_test, num_test_pos, num_test - num_test_pos))

    return (x_train, y_train), (x_test, y_test)

def slice_hdfs(x, y, window_size):#将X基于blk_id的序列每行进行window_size大小的窗口滑动，不足window_size大小的部分补#pad.
    results_data = []
    print("Slicing {} sessions, with window {}".format(x.shape[0], window_size))
    for idx, sequence in enumerate(x):
        seqlen = len(sequence)
        i = 0
        while (i + window_size) < seqlen:
            slice = sequence[i: i + window_size]
            results_data.append([idx, slice, sequence[i + window_size], y[idx]])
            i += 1
        else:
            slice = sequence[i: i + window_size]
            slice += ["#Pad"] * (window_size - len(slice))
            results_data.append([idx, slice, "#Pad", y[idx]])

    results_df = pd.DataFrame(results_data, columns=["SessionId", "EventSequence", "Label", "SessionLabel"])
    results_df.to_csv("日志序列滑动窗口.txt")
    print("Slicing done, {} windows generated".format(results_df.shape[0]))
    return results_df[["SessionId", "EventSequence"]], results_df["Label"], results_df["SessionLabel"]


#固定窗口大小，步长根据索引，提取特征
def load_BGL_feature(raw_data,event_data,window_size=40 ,stepping_size=10,train_ratio=0.8):
    """  TODO

    """
    label_data, time_data = raw_data[:, 0], raw_data[:, 1]
    start_index = 0
    end_index = 0


#提取bgl数据特征，窗口大小不固定，根据时间戳
def bgl_preprocess_data(para, raw_data, event_mapping_data):
    """ split logs into sliding windows, built an event count matrix and get the corresponding label

    Args:
    --------
    para: the parameters dictionary
    raw_data: list of (label, time)
    event_mapping_data: a list of event index, where each row index indicates a corresponding log

    Returns:
    --------
    event_count_matrix: event count matrix, where each row is an instance (log sequence vector)
    labels: a list of labels, 1 represents anomaly
    """

    # create the directory for saving the sliding windows (start_index, end_index), which can be directly loaded in future running
    if not os.path.exists(para['save_path']):
        os.mkdir(para['save_path'])
    log_size = raw_data.shape[0]
    print("log_size:",log_size)
    sliding_file_path = para['save_path']+'sliding_'+str(para['window_size'])+'h_'+str(para['step_size'])+'h.csv'

    #=============divide into sliding windows=========#
    start_end_index_list = [] # list of tuples, tuple contains two number, which represent the start and end of sliding time window
   # print("raw_data:",raw_data)
    label_data, time_data = raw_data[:,0], raw_data[:, 1]
    if not os.path.exists(sliding_file_path):
        # split into sliding window
        start_time = time_data[0]
        start_index = 0
        end_index = 0

        # get the first start, end index, end time
        for cur_time in time_data:
            if  cur_time < start_time + para['window_size']*3600:
                end_index += 1
                end_time = cur_time
            else:
                start_end_pair=tuple((start_index,end_index))
                #print("pair",start_end_pair)
                start_end_index_list.append(start_end_pair)
                break
        # move the start and end index until next sliding window
        #获得下一滑动窗口下的起始时间和起始下标
        while end_index < log_size:
            start_time = start_time + para['step_size']*3600
            end_time = end_time + para['step_size']*3600
            for i in range(start_index,end_index):
                if time_data[i] < start_time:
                    i+=1
                else:
                    break
            for j in range(end_index, log_size):
                if time_data[j] < end_time:
                    j+=1
                else:
                    break
            start_index = i
            end_index = j
            if j-i<5:
                #print("j-i ",j-i)
                continue
            else:
                start_end_pair = tuple((start_index, end_index))
                start_end_index_list.append(start_end_pair)
        #inst_number为一共的窗口数，也是样本数
        inst_number = len(start_end_index_list)
        print('there are %d instances (sliding windows) in this dataset\n'%inst_number)
        np.savetxt(sliding_file_path,start_end_index_list,delimiter=',',fmt='%d')
    else:
        print('Loading start_end_index_list from file')
        start_end_index_list = pd.read_csv(sliding_file_path, header=None).values
        inst_number = len(start_end_index_list)
        print('there are %d instances (sliding windows) in this dataset' % inst_number)

    # get all the log indexes in each time window by ranging from start_index to end_index
    expanded_indexes_list=[]
    for t in range(inst_number):
        index_list = []
        expanded_indexes_list.append(index_list)
    for i in range(inst_number):
        start_index = start_end_index_list[i][0]
        end_index = start_end_index_list[i][1]
        for l in range(start_index, end_index):
            #print("l:",l)
            expanded_indexes_list[i].append(l)

#    event_mapping_data = event_mapping_data.to_numpy().tolist()
    #print("event_mapping_data",event_mapping_data)
    event_num = len(list(set(event_mapping_data)))
    print('There are %d log events'%event_num)

    #=============get labels and event count of each sliding window =========#
    labels = []
    #*****事件计数矩阵 并给出事件序列data_instance*******
    data_instance=pd.DataFrame(columns=["EventSequence","Label"])#事件序列矩阵

    event_count_matrix = np.zeros((inst_number,event_num+1))
    print("event_mapping",len(event_mapping_data))
    print("expend_list:",len(expanded_indexes_list))
    pd.DataFrame(expanded_indexes_list).to_csv("../data/BGL/expend_list",columns=None)
    "generating data_instances and CountMatrix"
    for j in tqdm(range(inst_number)):
        #print(j)
        label = 0   #0 represent success, 1 represent failure
        EventSequence=[]
        #print("len:", len(expanded_indexes_list[j]))
        #print(expanded_indexes_list[inst_number-1])
        for k in expanded_indexes_list[j]:
            #print(k)
            event_index = event_mapping_data[k]
            #print("event_index",event_index)
            EventSequence.append(event_index)
            event_count_matrix[j, event_index] += 1
            if label_data[k]:
                label = 1
                continue
        data_instance.loc[j]=[EventSequence,label]
        #print("EventSequence:",EventSequence)

        labels.append(label)
    #print("data_instances:",data_instance)
    data_instance.to_csv("../data/BGL/data_instances.csv",index=False)
    assert inst_number == len(labels)
    print("Among all instances, %d are anomalies"%sum(labels))
    assert event_count_matrix.shape[0] == len(labels)
    print("event_count:",event_count_matrix.shape)
    print("labels:",len(labels))
    return event_count_matrix, labels
#将hdfs以块block为单位，块整体都在训练集里，按data_instance进行划分
def X_train_extract(data_dir="../data/HDFS",block_num=100000,train_ratio=0.5,normal_num=5000,normal_abnormal_ratio=1,file="../data/HDFS/data_instances.csv",template="../data/HDFS/HDFS.log_templates.csv",data_type='HDFS'):
    if data_type == 'HDFS':
        Event_seq = pd.read_csv(file, engine='c',
                                na_filter=False, memory_map=True)
        #根据日志分析的事件模板排序template记录Events种类和序号，用Events序号代替解析后的日志模板ID
        Events_list=pd.read_csv(template, engine='c',
                                na_filter=False, memory_map=True)
        Events = Events_list["EventId"].tolist()
        Events.insert(0,"#")
        #Events.append("#")
        length = block_num
        max_length=len(Event_seq)
        if length>len(Event_seq):
            length=len(Event_seq)
            print("max blocks:",length)
        mid = int(length * train_ratio)
        # Event_seq将原事件ID按mapping替换成数字
        # 数据前一半的正常序列作为训练，后一半的正常+异常作测试
        for i in range(max_length):
            strs = Event_seq.at[i, "EventSequence"].replace('[', '').replace(']', '').replace('\'', '').replace(',',
                                                                                                                ' ').split()
            # print("strs:",strs)
            #用template生成的Events序号代替解析后的日志模板ID
            for j, ch in enumerate(strs):
                if ch in Events:
                    strs[j] = str(Events.index(ch))
                else:
                    print("unknown event!")
            Event_seq.at[i, "EventSequence"] = ' '.join(strs)
        print("事件总数:", len(Events))
        print("事件ID顺序", Events)

        Event_seq.to_csv("../data/HDFS/hdfs.csv")
        #前length条随机重排！！！
        Event_last=Event_seq[length:]
        pd.DataFrame(Event_last[['EventSequence', 'Label']]).to_csv("../data/HDFS/MLog_log_last.csv", index=None)
        print("last length:",len(Event_last))
        Event_seq=Event_seq[:length]
        Event_seq = Event_seq.sample(frac=1).reset_index(drop=True)
        Event_seq=Event_seq.reset_index(drop=True)
        Event_seq.to_csv("../data/HDFS/hdfs_shuffled.csv")

        df = Event_seq[:block_num][['EventSequence', 'Label']]
        normal_df = pd.DataFrame(pd.DataFrame(df).loc[df['Label'] == 0])
        abnormal_df = pd.DataFrame(pd.DataFrame(df).loc[df['Label'] == 1])

        #去除重复的
        normal_df1=normal_df[:normal_num]
        normal_df2=normal_df.drop(normal_df1.index)

        test_df=pd.DataFrame(pd.concat([pd.DataFrame(normal_df2),pd.DataFrame(abnormal_df)]))

        print("无监督的HDFS：")
        print("train_sequence:", len(normal_df1), " train_normal_df:", len(normal_df1), " train_abnormal_df:",
              0)
        print("test_sequence:", len(test_df), " test_normal_df:", len(normal_df2), " test_abnormal_df:",
              len(abnormal_df))

        with open("../data/HDFS/hdfs_train", 'w') as wf:
            for i,row in normal_df1.iterrows():
                    wf.write(row["EventSequence"] + "\n")

        with open("../data/HDFS/hdfs_test_normal", 'w') as wf1:
            with open("../data/HDFS/hdfs_test_abnormal", 'w') as wf2:

                  for i,row in normal_df2.iterrows():
                        wf1.write(row["EventSequence"] + "\n")
                  for i,row in abnormal_df.iterrows():
                        wf2.write(row["EventSequence"] + "\n")
        with open("../data/HDFS/Events_mapping", 'w') as wf3:
            wf3.write(str(Events))
        print("keys:", Event_seq.columns)


        abnormal_num=int(normal_num*(1.0/normal_abnormal_ratio))
        df = Event_seq[:][['EventSequence', 'Label']]
        normal_df= pd.DataFrame(df).loc[df['Label'] == 0]
        abnormal_df = pd.DataFrame(df).loc[df['Label'] == 1]
        abnormal_df1=abnormal_df[:abnormal_num]
        normal_df1=normal_df[:normal_num]
        abnormal_df2=abnormal_df[abnormal_num:]
        normal_df2 = normal_df[abnormal_num * normal_abnormal_ratio:]
        train_df = pd.concat([pd.DataFrame(normal_df1), pd.DataFrame(abnormal_df1)])
        predict_df=pd.concat([pd.DataFrame(normal_df2), pd.DataFrame(abnormal_df2)])
        print("有监督的HDFS：")
        print("train_sequence:", len(train_df), " train_normal_df:", len(normal_df1), " train_abnormal_df:",len(abnormal_df1))
        print("test_sequence:", len(predict_df), " test_normal_df:", len(normal_df2)," test_abnormal_df:", len(abnormal_df2))

        train_df.columns=["Sequence", 'label']
        predict_df.columns=["Sequence", 'label']
        pd.DataFrame(train_df).to_csv("../data/HDFS/MLog_log_train.csv", index=None)
        pd.DataFrame(train_df).to_csv("../data/HDFS/MLog_log_valid.csv", index=None)
        pd.DataFrame(predict_df).to_csv("../data/HDFS/MLog_log_test.csv", index=None)
        """
        df1 = Event_seq[0:mid][['EventSequence', 'Label']]
        df1.columns = ["Sequence", 'label']
        normal_df = pd.DataFrame(df1).loc[df1['label'] == 0]

        abnormal_df = pd.DataFrame(df1).loc[df1['label'] == 1]
        abnormal_len = len(abnormal_df)

        #normal_df = normal_df.sample(n=int(abnormal_len * (normal_ratio/(1-normal_ratio))), random_state=1)
        train_df = pd.concat([pd.DataFrame(normal_df), pd.DataFrame(abnormal_df)])
        print("train_sequence:",len(train_df)," train_normal_df:", len(normal_df)," train_abnormal_df:", len(abnormal_df))

        pd.DataFrame(train_df).to_csv("../data/HDFS/MLog_log_train.csv", index=None)
        pd.DataFrame(train_df).to_csv("../data/HDFS/MLog_log_valid.csv", index=None)
        df2 = Event_seq[mid:length][['EventSequence', 'Label']]
        df2.columns = ["Sequence", 'label']
        print("test_sequence:", len(df2), " test_normal_df:", len(pd.DataFrame(df2).loc[df2['label'] == 0]), " test_abnormal_df:",len(pd.DataFrame(df2).loc[df2['label'] == 1]))
        pd.DataFrame(df2).to_csv("../data/HDFS/MLog_log_test.csv", index=None)"""
        # pd.DataFrame(df2).to_csv("../data/HDFS/MLog_log_valid.csv", index=None)
        return Events


    elif data_type == 'BGL'or data_type=='Thunderbird':
        """
        if data_type=='BGL':
            file="../data/BGL/data_instances.csv"
        elif data_type=='Thunderbird':
            file="../data/Thunderbird/data_instances.csv"
        """

        Event_seq = pd.read_csv(file, engine='c',
                                na_filter=False, memory_map=True)
        # 根据日志分析的事件模板排序template记录Events种类和序号，用Events序号代替解析后的日志模板ID
        Events_list = pd.read_csv(template, engine='c',
                                  na_filter=False, memory_map=True)
        Events = Events_list["EventId"].tolist()
        Events.insert(0, "#")
        #print("mapping2:",Events)
        # Events.append("#")
        length = block_num
        max_length = len(Event_seq)
        if length > len(Event_seq):
            length = len(Event_seq)
            print("max blocks:", length)
        mid = int(length * train_ratio)
        # Event_seq将原事件ID按mapping替换成数字
        # 数据前一半的正常序列作为训练，后一半的正常+异常作测试
        for i in range(max_length):
            strs = Event_seq.at[i, "EventSequence"].replace('[', '').replace(']', '').replace('\'', '').replace(',', ' ').split()
            # print("strs:",strs)
            """
            # 用template生成的Events序号代替解析后的日志模板ID
            for j, ch in enumerate(strs):
                if ch in Events:
                    strs[j] = str(Events.index(ch))
                else:
                    print("unknown event!")
                    """
            Event_seq.at[i, "EventSequence"] = ' '.join(strs)

        print("事件总数:", len(Events))
        print("事件ID顺序", Events)

        Event_seq.to_csv("../data/"+data_type+"/"+data_type+".csv")
        Event_last = Event_seq[length:]
        pd.DataFrame(Event_last[['EventSequence', 'Label']]).to_csv("../data/"+data_type+"/MLog_log_last.csv", index=None)
        print("last length:", len(Event_last))
        Event_seq = Event_seq[:length]
        # 前length条随机重排！！！
        Event_seq = Event_seq.sample(frac=1).reset_index(drop=True)
        Event_seq = Event_seq.reset_index(drop=True)
        Event_seq.to_csv("../data/"+data_type+"/"+data_type+"_shuffled.csv")

        df = Event_seq[:][['EventSequence', 'Label']]
        total_num=len(df)

        normal_df = pd.DataFrame(pd.DataFrame(df).loc[df['Label'] == 0])
        abnormal_df = pd.DataFrame(pd.DataFrame(df).loc[df['Label'] == 1])

        # 去除重复的
        normal_df1 = normal_df[:int(len(normal_df)*train_ratio)]
        normal_df2 = normal_df.drop(normal_df1.index)
        abnormal_df1=abnormal_df[:int(len(abnormal_df)*train_ratio)]
        abnormal_df2=abnormal_df.drop(abnormal_df1.index)

        test_df = pd.DataFrame(pd.concat([pd.DataFrame(normal_df2), pd.DataFrame(abnormal_df2)]))

        print("无监督的"+data_type+"：")
        print("train_sequence:", len(normal_df1)+len(abnormal_df1), " train_normal_df:", len(normal_df1), " train_abnormal_df:",
              len(abnormal_df1))
        print("test_sequence:", len(test_df), " test_normal_df:", len(normal_df2), " test_abnormal_df:",
              len(abnormal_df2))

        with open("../data/"+data_type+"/"+data_type+"_train", 'w') as wf:
            for i, row in normal_df1.iterrows():
                wf.write(row["EventSequence"] + "\n")
        with open("../data/"+data_type+"/"+data_type+"_train_abnormal", 'w') as wf:
            for i, row in abnormal_df1.iterrows():
                print(row["EventSequence"])
                wf.write(row["EventSequence"] + "\n")
        with open("../data/"+data_type+"/"+data_type+"_test_normal", 'w') as wf1:
            with open("../data/"+data_type+"/"+data_type+"_test_abnormal", 'w') as wf2:

                for i, row in normal_df2.iterrows():
                    wf1.write(row["EventSequence"] + "\n")
                for i, row in abnormal_df2.iterrows():
                    wf2.write(row["EventSequence"] + "\n")
        with open("../data/"+data_type+"/Events_mapping", 'w') as wf3:
            wf3.write(str(Events))
        print("keys:", Event_seq.columns)

        abnormal_num = int(normal_num * (1.0 / normal_abnormal_ratio))
        df = Event_seq[:][['EventSequence', 'Label']]
        normal_df = pd.DataFrame(df).loc[df['Label'] == 0]
        abnormal_df = pd.DataFrame(df).loc[df['Label'] == 1]
        abnormal_df1 = abnormal_df[:abnormal_num]
        normal_df1 = normal_df[:normal_num]
        abnormal_df2 = abnormal_df[abnormal_num:]
        normal_df2 = normal_df[abnormal_num * normal_abnormal_ratio:]
        train_df = pd.concat([pd.DataFrame(normal_df1), pd.DataFrame(abnormal_df1)])
        predict_df = pd.concat([pd.DataFrame(normal_df2), pd.DataFrame(abnormal_df2)])
        print("有监督的"+data_type+"：")
        print("train_sequence:", len(train_df), " train_normal_df:", len(normal_df1), " train_abnormal_df:",
              len(abnormal_df1))
        print("test_sequence:", len(predict_df), " test_normal_df:", len(normal_df2), " test_abnormal_df:",
              len(abnormal_df2))

        train_df.columns = ["Sequence", 'label']
        predict_df.columns = ["Sequence", 'label']
        pd.DataFrame(train_df).to_csv("../data/"+data_type+"/MLog_log_train.csv", index=None)
        pd.DataFrame(train_df).to_csv("../data/"+data_type+"/MLog_log_valid.csv", index=None)
        pd.DataFrame(predict_df).to_csv("../data/"+data_type+"/MLog_log_test.csv", index=None)

        return Events

#将初步事件序列data_instance 按需要比例划分整理到 hdfs_train、hdfs_test,robust_train,bgl_train等

def BGL_data(file="../data/BGL/1000000_BGL.log_structured.csv",template="../data/BGL/1000000_BGL.log_templates.csv"):
    read_file=pd.read_csv(file)
    mapping=getMapping(template)
    mapping.insert(0, "#")
    #替换标签为0 1 ,EventID替换为数字
    read_file.Label=read_file.Label.map(lambda x: 0 if x=='-' else 1)
    read_file.EventId=read_file.EventId.map(lambda x: mapping.index(x))


    data_out=pd.concat([read_file["Label"],read_file["Timestamp"]],axis=1)
    print("data_out:",data_out)
    Events_mapping=read_file["EventId"].values.tolist()
    #print("Events:",Events_mapping)
    return data_out.values,Events_mapping

def get_BGL_template(template="../data/BGL/1000000_BGL.log_templates.csv"):
    read_file = pd.read_csv(template)
    templates=dict()
    for i,row in read_file.iterrows():
        templates[i]=row["EventTemplate"]
    return templates

def getMapping(file):
    read_file = pd.read_csv(file)
    Events_mapping=read_file["EventId"].tolist()
    return  Events_mapping

if __name__=='__main__':
    #HDFS:
    """
    (x_train, y_train), (x_test, y_test) = load_HDFS("../data/HDFS/HDFS.log_structured.csv",
                                                                label_file="../data/HDFS/anomaly_label.csv",
                                                                window='session',
                                                                train_ratio=0.7,
                                                                split_type='uniform',
                                                                save_csv=True)#加载数据，提取每个blkID的事件集，保存至/HDFS/data_instance
    """



    #Events_mapping=X_train_extract(block_num=100000,train_ratio=0.5,normal_num=6000,normal_abnormal_ratio=3)#与X_train_fit不同，该函数仅仅作用于hdfs,用于完整提取一个blockid所有日
    #Events_mapping = X_train_extract(block_num=1000000, train_ratio=0.5, normal_num=6000,normal_abnormal_ratio=1)  # 与X_train_fit不同，该函数仅仅作用于hdfs,用于完整提取一个blockid所有日


    Events_Bert_vectors("../data/HDFS/HDFS.log_templates.csv",tf_idf=0,IDF=1,scalermin=0.7,scalermax=1,events_semantic="../data/HDFS/events_semantic.json")


    #BGL:
    """
    para = dict()
    para["save_path"] = "../data/BGL/"
    para["step_size"] = 0.0001
    para["window_size"] = 0.0001

    #Events_mapping指事件ID序列
    BGL_data, Events_mapping = BGL_data(file="../data/BGL/BGL.log_structured.csv",template="../data/BGL/BGL.log_templates.csv")
    print("len",BGL_data.size)
    print("Events_mapping:",len(Events_mapping))
    Events_count, labels = bgl_preprocess_data(para, BGL_data, Events_mapping)
    print("mapping",Events_mapping)
    """
    #X_train_extract(block_num=100000,data_type='BGL',train_ratio=0.5,file="../data/BGL/data_instances.csv",template="../data/BGL/BGL.log_templates.csv",normal_num=5000,normal_abnormal_ratio=1)

    ##事件ID映射根据 templates文件

    #templates=Detect_Events("../data/BGL/BGL.log_templates.csv")
    #Events_vectors(templates,events_semantic="../data/BGL/events_semantic.json",tf_idf=1,template="../data/BGL/BGL.log_templates.csv")
    #Events_Bert_vectors("../data/BGL/BGL.log_templates.csv",tf_idf=0,IDF=1,scalermin=0.7 ,scalermax=1,events_semantic="../data/BGL/events_semantic.json")





