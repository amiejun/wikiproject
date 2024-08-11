'''
目标：将文本数据转换为向量，并用向量建立一个faiss索引。
关键点：
  文本向量化：使用bert模型将文本转化为向量
  faiss索引：使用faiss库快速检索相似项
  批处理：为提高效率，文本数据以批次方式进行向量化

  代码逻辑：
  1从csv文件读取文本数据。
  2将文本数据转化为向量
  3使用向量建立faiss索引
  4保存索引和URL映射到文件



'''

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel


class Indexer:
    def __init__(self, model_name='uer/roberta-base-finetuned-chinanews-chinese', batch_size=8):#复制模型名称填入
        # 检查是否可以使用CUDA
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.tokenizer = BertTokenizer.from_pretrained(model_name)#与模型的编码方式要一致
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.index = faiss.IndexFlatL2(768)  # BERT的向量大小为768
        self.batch_size = batch_size
        self.url_mapping = []  # 存储每个向量对应的URL




    # 这个函数可以认为是transformer文本向量化的一个标准写法
    def texts_to_vectors(self, texts):
        vectors = []#预制空列表
        for i in tqdm(range(0, len(texts), self.batch_size)):#qtdm为进度条
            batch_texts = texts[i:i + self.batch_size].tolist()#截取文本
            inputs = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)#把文本变成了数字id
            inputs = {k: v.to(self.device) for k, v in inputs.items()}#把输入转化到指定设备上
            outputs = self.model(**inputs)#把数据喂到模型里面去拿到他的表征 模型来自38line
            batch_vectors = outputs.pooler_output.detach().cpu().numpy()#把从output中拿出的pooler_output转化成numpy
            vectors.extend(batch_vectors) #追加结果至vectors中
        return np.array(vectors)

    #这个函数的意思是将向量化的结果刷到index中
    def add_to_index(self, texts, urls):
        vectors = self.texts_to_vectors(texts)
        self.index.add(vectors)
        self.url_mapping.extend(urls) #记录索引和url的映射关系 将新洗出来的url加入到原本index的集里面

    def save_index_and_mapping(self, index_path, mapping_path):
        faiss.write_index(self.index, index_path)
        with open(mapping_path, 'w', encoding='utf-8') as f:
            for url in self.url_mapping:
                f.write(url + '\n')   #写入id-url的映射

    def build_index_from_csv(self, csv_file_path):
        df = pd.read_csv(csv_file_path)#[:100000] # 为了演示这里只选取了1000条数据
        self.add_to_index(df['content'], df['url'])

if __name__=="__main__":
    # 示例用法
    indexer = Indexer()
    indexer.build_index_from_csv('./data/wiki_zh.csv')  # 假设CSV文件路径
    indexer.save_index_and_mapping('./data/wiki_zh.index', './data/wiki_map.txt')
