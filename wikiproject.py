'''

将使用wiki2019zh数据集，构建一个搜索索引，并通过一个基于streamlit的web应用进行搜索。

项目结构：
1数据预处理（process_data):处理wiki数据
2索引器（indexer）：用于构建搜索引擎
3搜索器（searcher）：用于执行搜索操作
3web应用（app）：用户界面，用于接收查询和展示搜索结果


运行项目：


1设置环境：
确保安装了所有的依赖，包括faiss-cpu.transformers,panda和sreamlit
2 数据预处理：
预处理wiki数据
3构建索引
首先运行indexer.py 来构建搜索索引。
4启动web应用
运行app.py来启动streamlit

在终端中输入指令：

streamlit run app.py
5搜索查询：
在web应用中输入查询，查看搜索结果。

运行顺序：2-3-4
'''
