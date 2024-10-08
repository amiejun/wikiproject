"""
提供界面，允许用户输入搜索查询并显示结果
关键点：
streamlit：用于快速构建web应用。
交互：接收用户输入并展示结果。

代码逻辑：
1.初始化streamlit应用
2 创建文本输入框用于接受查询
3 调用搜索器获取结果。
4展示搜索结果。


"""
import streamlit as st
from searcher import Searcher

# 初始化搜索器 - 替换为你的索引和映射文件路径
searcher = Searcher('./data/wiki_zh.index', './data/wiki_map.txt')

st.title('Wiki搜索引擎')

query = st.text_input('输入你的查询', '')

if query:
    results = searcher.search(query, k=5)  # 可以调整 k 的值来改变返回的结果数量
    if results:
        st.subheader('搜索结果')
        for url in results:
            st.write(url)
    else:
        st.write('没有找到相关结果')
