'''
加载之前创建的faiss索引和URL映射，以便对用户进行搜索。

关键点：
加载索引：从文件中加载faiss索引
查询处理：将用户查询转换为向量。
检索匹配：使用索引找到最相似的文本项。

代码逻辑：
1加载faiss索引和URL映射。
2将用户查询转换为向量。
3在索引中搜索最相似的项
4返回相应的URLs




'''

import faiss
import numpy as np
from transformers import BertTokenizer, BertModel

class Searcher:
    def __init__(self, index_path, mapping_path, model_name='uer/roberta-base-finetuned-chinanews-chinese'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.index = faiss.read_index(index_path)
        self.urls = self.load_url_mapping(mapping_path)

    def load_url_mapping(self, mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as file:
            urls = [line.strip() for line in file]
        return urls

    def query_to_vector(self, query):
        inputs = self.tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = self.model(**inputs)
        vector = outputs.pooler_output.detach().numpy()
        return vector

    def search(self, query, k=10):
        query_vector = self.query_to_vector(query) #把quary变向量
        _, I = self.index.search(query_vector, k)#走faiss，拿到index
        results = [self.urls[i] for i in I[0]]  # 根据索引找到对应的URL
        return results

if __name__=="__main__":
    # 示例用法
    searcher = Searcher('./data/wiki_zh.index', './data/wiki_map.txt')
    search_results = searcher.search('孙悟空', k=5)
    for url in search_results:
        print(url)
