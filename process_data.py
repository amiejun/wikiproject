'''

将wiki数据转化为csv格式。

关键点：
wiki数据集理解：理解wiki数据集的组织形式
单个数据转换脚本：转换为csv文件
批量转换脚本：批量转换所有数据为csv格式

代码逻辑
1批量遍历wiki数据集下所有文件
2将每个文件转换为csv格式，并实时追加到csv文件中

'''

import json
import csv
import os

def process_line_to_csv(json_line,csv_writer):
    try:
        article=json.loads(json_line)
        content=article.get('text','')
        url=article.get('url','')
        csv_writer.writerow([content, url])
    except json.JSONDecodeError:
        print('警告：无法解析的行')

def process_file(json_file_path,csv_writer):
    with open(json_file_path,'r',encoding='utf-8') as file:
        for line in file:
            process_line_to_csv(line,csv_writer)


def process_directory(directory_path,output_file,num_file=100):
    with open(output_file,'w',encoding='utf-8',newline='') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow(['content','url']) #写入标题行

        file_count=0

        for root,dirs,files in os.walk(directory_path):
            for file in files:
                file_count +=1
                json_file_path=os.path.join(root,file)
                process_file(json_file_path,writer)
                print(f'Iter:{file_count}/{num_file}Process File:{json_file_path}')
                if file_count>num_file:
                    return
if __name__=='__main__':
    wiki_directory='./data/wiki_zh' #wiki2019zh 数据集的目录路径
    output_csv='./data/wiki_zh.csv' #输出的csv路径
    num_file=100 #处理的数据数量
    process_directory(wiki_directory,output_csv,num_file)

