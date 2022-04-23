import torch
import numpy as np
import pandas as pd
from pandas import DataFrame

from pytorch_pretrained_bert import BertTokenizer,BertModel,BertForMaskedLM



tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def normalization(data):
    _range = np.max(data) - np.min(data)
    if np.max(data)==np.min(data):
        return (data - np.min(data)+0.5) / (_range+1)+0.5
    return (data - np.min(data)) / _range+0.5
def word2vector(str):

    #out_list=[]
    # Tokenized input

    text = "[CLS]"+str+"[SEP]"
        #1.分词token
    tokenized_text = tokenizer.tokenize(text)
    print("token:",len(tokenized_text),tokenized_text)
        #print(tokenized_text)
        #2.给分词对应词表查找标签
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    #for tup in zip(tokenized_text, indexed_tokens):
      #print (tup)
        #3.属句标签
    segments_ids = [1] * len(tokenized_text)
    #print (segments_ids)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    # Load pre-trained model (weights)
    #model = BertModel.from_pretrained('bert-base-uncased')
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    # Predict hidden states features for each layer
    with torch.no_grad():
      encoded_layers, _ = model(tokens_tensor, segments_tensors)
      print ("out:",len(encoded_layers),len(encoded_layers[0]),len(encoded_layers[0][0]),len(encoded_layers[0][0][0]))
      sentence_embedding = torch.mean(encoded_layers[11], 1)
    #out_list.append(list(np.array(sentence_embedding[0])))

    #print("out_shape:",len(out_list),",",len(out_list[0]))
      out_array = sentence_embedding[0].numpy().tolist()
    #print("out_array.shape",len(out_array))
    return out_array
def word2vector2(str,template):#选取后四层，去掉cls和sep
    # out_list=[]
    # Tokenized input

    text = "[CLS]" + str + "[SEP]"
    # 1.分词token
    tokenized_text = tokenizer.tokenize(text)[4:-3]
    print("token:", len(tokenized_text), tokenized_text)
    # print(tokenized_text)
    # 2.给分词对应词表查找标签
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # for tup in zip(tokenized_text, indexed_tokens):
    # print (tup)
    # 3.属句标签
    segments_ids = [1] * len(tokenized_text)
    # print (segments_ids)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    # Load pre-trained model (weights)
    # model = BertModel.from_pretrained('bert-base-uncased')
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
        print("out:", len(encoded_layers), len(encoded_layers[0]), len(encoded_layers[0][0]),
              len(encoded_layers[0][0][0]))
        print("type:",type(encoded_layers),type(encoded_layers[0]))
        encoded_layers=torch.stack(encoded_layers[8:])#list转tensor,取bert 12层中的最后四层
        print("layers:",len(encoded_layers))
        sentence_embedding = torch.mean(encoded_layers, 0)
        print("shape",sentence_embedding.numpy().shape)
        sentence_embedding = torch.mean(sentence_embedding, 1)
        print("shape2", sentence_embedding.numpy().shape)
        # out_list.append(list(np.array(sentence_embedding[0])))

        # print("out_shape:",len(out_list),",",len(out_list[0]))
        out_array = sentence_embedding[0].numpy().tolist()
    # print("out_array.shape",len(out_array))
    return out_array

def pre_handle1(sentence):  # 对句子进行预处理，提取词干,去掉符号和数字,
    str_list = list(sentence)  # 字符串转化为列表，单个字符
    # print(str_list)
    for i, char in enumerate(str_list):  # 只保留字母，别的符号和数字用空格代替
        if char >= 'a' and char <= 'z':
            continue
        elif char >= '0' and char <= '9':
            if i > 0 and str_list[i - 1] >= 'a' and str_list[i - 1] <= 'z':
                str_list.insert(i, ' ')
            if i + 1 < len(str_list) and str_list[i + 1] >= 'a' and str_list[i + 1] <= 'z':
                str_list.insert(i + 1, ' ')
            continue
        elif char == ' ' or char=='=' or char==':':
            continue
        else:
            str_list[i] = ' '
    return ''.join(str_list).strip()

def pre_handle2(sentence):  # 对句子进行预处理，提取词干,去掉符号和数字,
    str_list = list(sentence)  # 字符串转化为列表，单个字符
    # print(str_list)
    for i, char in enumerate(str_list):  # 只保留字母，别的符号和数字用空格代替
        if char >= 'a' and char <= 'z' or  char >= 'A' and char <= 'Z':
            continue

        elif char >= '0' and char <= '9':
            if i > 0 and str_list[i - 1] >= 'a' and str_list[i - 1] <= 'z':
                str_list.insert(i, ' ')
            if i + 1 < len(str_list) and str_list[i + 1] >= 'a' and str_list[i + 1] <= 'z':
                str_list.insert(i + 1, ' ')
            continue



        elif char==' ' or char=='*' or char=='=' or char==':' or char==',' or char=='.':
            continue
        else:
            str_list[i]=' '
    # 将列表组合成字符串
    sstr = ''.join(str_list)
    return ''.join(str_list).strip()

def word2vector3(str,template,tf_idf):
    # out_list=[]
    # Tokenized input
    print("str",str)

    temp=pre_handle1(str)
    if temp!="" and temp!=" ":
        str=temp

    print("str_new",str )
    text = "[CLS]" + str + "[SEP]"
    # 1.分词token
    tokenized_text = tokenizer.tokenize(text)[4:-3]
    print("token:", len(tokenized_text), tokenized_text)
    # print(tokenized_text)
    # 2.给分词对应词表查找标签
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # for tup in zip(tokenized_text, indexed_tokens):
    # print (tup)
    # 3.属句标签
    segments_ids = [1] * len(tokenized_text)
    # print (segments_ids)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    # Load pre-trained model (weights)
    # model = BertModel.from_pretrained('bert-base-uncased')
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()
    tf_idf_vect = []
    totallogs = 0
    templates = pd.read_csv(template, engine='c')
    for index, row in templates.iterrows():
        totallogs += int(row['Occurrences'])
    if tf_idf == 1:
      for word in tokenized_text:
        tf = float(str.count(word)) / len(str)
        idf0 = 0
        for index, row in templates.iterrows():
            if word in row["EventTemplate"]:
                # print("Event:",row["EventTemplate"])
                idf0 += row['Occurrences']
        idf = np.log(float(totallogs) / (idf0 + 1))
        print("idf:", idf)
        tf_idf_vect.append(tf * idf)
    #print("total logs:", totallogs)
    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
        print("out:", len(encoded_layers), len(encoded_layers[0]), len(encoded_layers[0][0]),
              len(encoded_layers[0][0][0]))
        print("type:",type(encoded_layers),type(encoded_layers[0]))

        encoded_layers=torch.stack(encoded_layers[8:])#list转tensor,取bert 12层中的最后四层
        print("layer——shape:",len(encoded_layers),len(encoded_layers[0]),len(encoded_layers[0][0]),len(encoded_layers[0][0][0]))

        sentence_embedding = torch.mean(encoded_layers, 0)[0].numpy()
        print("shape",sentence_embedding.shape)
        print("tf_idf_vect:", tf_idf_vect)

        if tf_idf == 1:
            if len(tf_idf_vect) != 0:
                tf_idf_vect = normalization(tf_idf_vect)
            print("tf_idf_vect_normalization:", tf_idf_vect)


            print("sentence_len:", len(sentence_embedding))
            for i in range(len(sentence_embedding)):
                # print(sentence_vector[i])
                sentence_embedding[i] = tf_idf_vect[i] * sentence_embedding[i]
        print("shape2", sentence_embedding.shape)
        # out_list.append(list(np.array(sentence_embedding[0])))

        # print("out_shape:",len(out_list),",",len(out_list[0]))
        tensor = torch.tensor(sentence_embedding)
        # print(tensor)
        out = torch.mean(tensor, dim=0)
        out_array = out.numpy().tolist()
    # print("out_array.shape",len(out_array))
    return out_array

if __name__=='__main__':

    str="generating a root user"
    print(word2vector3(str,'../../data/BGL/BGL.log_templates.csv',1))



