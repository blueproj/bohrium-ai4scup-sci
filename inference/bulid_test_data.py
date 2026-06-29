import jsonlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)

import os
import re
from functools import partial

from langchain.text_splitter import MarkdownHeaderTextSplitter,MarkdownTextSplitter, MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter, CharacterTextSplitter
from FlagEmbedding import BGEM3FlagModel,FlagReranker

def task2_rag(md_path, query):
    with open(md_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 寻找 # References 标题的位置
    # pattern = r'(?i)^# References\b'  # (?i) 忽略大小写匹配
    # # 使用正则表达式查找 References 部分
    # match = re.search(pattern, content, re.MULTILINE)
    # if match:
    #     # 切分并保留 # References 之前的内容
    #     content_before_references = content[:match.start()]
    # else:
    #     # 如果未找到 # References，则保留全部内容
    #     content_before_references = content
    # print(content_before_references)
    # print(content_before_references)
    # print('++++++++++++++++++++++++++++++++++')
    # 将文本中的img路径都删除
    pattern = "!\[.*?\]\(images/[^/]*?\.jpg\)"
    text = re.sub(pattern, '', content)

    text = re.sub(r' +', ' ', text)
    # print(text)
    # print('++++++++++++++++++++++++++++++++++')
    # 定义正则表达式，匹配连续15个或更多大写字母
    text = re.sub(r'[A-Z]{15,}', '', text)
    text = text.replace('\n\n', '\n')
    # print(text)
    # print('++++++++++++++++++++++++++++++++++')

    results = []
    
    # headers_to_split_on = [
    # ("#", "Header 1"),
    # ("##", "Header 2"),
    # ("###", "Header 3"),
    # ]
    # markdown_splitter = MarkdownHeaderTextSplitter(
    #     headers_to_split_on=headers_to_split_on)
    # rs = markdown_splitter.split_text(text)
    

    chunk_size = 896
    chunk_overlap = 64
    # text_splitter = CharacterTextSplitter(
    #     separator=' ',chunk_size=chunk_size, chunk_overlap=chunk_overlap
    # )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n"," ", ""]
    )
    # Split
    docs = text_splitter.split_text(text)
    for doc in docs:
        # print(doc.page_content)
        # print('++++++++++++++++++++++++++++++++++')
        results.append(doc)

    embeddings_1 = emb_model.encode([query], 
                                batch_size=12, 
                                max_length=1024, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                )['dense_vecs']
    embeddings_2 = emb_model.encode(results)['dense_vecs']
    similarity = embeddings_1 @ embeddings_2.T

    # 找出前10大值对应的索引
    top10_indices = sorted(range(len(similarity[0])), key=lambda i: similarity[0][i], reverse=True)[:10]
    emb_top_10_results = [results[i] for i in top10_indices]

    if len(emb_top_10_results) < 2:
        return emb_top_10_results
    
    reranker_list = []
    for i in emb_top_10_results:
        reranker_list.append([query,i])
    scores = reranker_model.compute_score(reranker_list, normalize=True)

    top3_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2]

    final_top_3_results = [emb_top_10_results[i] for i in top3_indices]
    return final_top_3_results

def task5_rag(md_path, query):
    with open(md_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 寻找 # References 标题的位置
    pattern = r'(?i)^# References\b'  # (?i) 忽略大小写匹配
    # 使用正则表达式查找 References 部分
    match = re.search(pattern, content, re.MULTILINE)
    if match:
        # 切分并保留 # References 之前的内容
        content_before_references = content[:match.start()]
    else:
        # 如果未找到 # References，则保留全部内容
        content_before_references = content
    # print(content_before_references)
    # print(content_before_references)
    # print('++++++++++++++++++++++++++++++++++')
    # 将文本中的img路径都删除
    pattern = "!\[.*?\]\(images/[^/]*?\.jpg\)"
    text = re.sub(pattern, '', content_before_references)

    # 将连续的多个空格替换为单空格
    text = re.sub(r' +', ' ', text)
    # print(text)
    # print('++++++++++++++++++++++++++++++++++')
    # 定义正则表达式，匹配连续15个或更多大写字母
    text = re.sub(r'[A-Z]{15,}', '', text)

    text = text.replace('\n\n', '\n')
    # print(text)
    # print('++++++++++++++++++++++++++++++++++')

    results = []
    
    # headers_to_split_on = [
    # ("#", "Header 1"),
    # ("##", "Header 2"),
    # ("###", "Header 3"),
    # ]
    # markdown_splitter = MarkdownHeaderTextSplitter(
    #     headers_to_split_on=headers_to_split_on)
    # rs = markdown_splitter.split_text(text)
    

    chunk_size = 768
    chunk_overlap = 64
    # text_splitter = CharacterTextSplitter(
    #     separator=' ',chunk_size=chunk_size, chunk_overlap=chunk_overlap
    # )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=[" ", ""]
    )
    # Split
    docs = text_splitter.split_text(text)
    for doc in docs:
        # print(doc.page_content)
        # print('++++++++++++++++++++++++++++++++++')
        results.append(doc)

    embeddings_1 = emb_model.encode([query], 
                                batch_size=12, 
                                max_length=1024, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                )['dense_vecs']
    embeddings_2 = emb_model.encode(results)['dense_vecs']
    similarity = embeddings_1 @ embeddings_2.T

    # 找出前10大值对应的索引
    top10_indices = sorted(range(len(similarity[0])), key=lambda i: similarity[0][i], reverse=True)[:10]
    emb_top_10_results = [results[i] for i in top10_indices]

    if len(emb_top_10_results) < 2:
        return emb_top_10_results
    
    reranker_list = []
    for i in emb_top_10_results:
        reranker_list.append([query,i])
    scores = reranker_model.compute_score(reranker_list, normalize=True)

    top3_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:1]

    final_top_3_results = [emb_top_10_results[i] for i in top3_indices]
    return final_top_3_results

def task6_rag(md_path, query):
    with open(md_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 寻找 # References 标题的位置
    # pattern = r'(?i)^# References\b'  # (?i) 忽略大小写匹配
    # # 使用正则表达式查找 References 部分
    # match = re.search(pattern, content, re.MULTILINE)
    # if match:
    #     # 切分并保留 # References 之前的内容
    #     content_before_references = content[:match.start()]
    # else:
    #     # 如果未找到 # References，则保留全部内容
    #     content_before_references = content
    # print(content_before_references)
    # print(content_before_references)
    # print('++++++++++++++++++++++++++++++++++')
    # 将文本中的img路径都删除
    pattern = "!\[.*?\]\(images/[^/]*?\.jpg\)"
    text = re.sub(pattern, '', content)

    text = re.sub(r' +', ' ', text)
    # print(text)
    # print('++++++++++++++++++++++++++++++++++')
    # 定义正则表达式，匹配连续15个或更多大写字母
    text = re.sub(r'[A-Z]{15,}', '', text)
    text = text.replace('\n\n', '\n')
    # print(text)
    # print('++++++++++++++++++++++++++++++++++')

    results = []
    
    # headers_to_split_on = [
    # ("#", "Header 1"),
    # ("##", "Header 2"),
    # ("###", "Header 3"),
    # ]
    # markdown_splitter = MarkdownHeaderTextSplitter(
    #     headers_to_split_on=headers_to_split_on)
    # rs = markdown_splitter.split_text(text)
    

    chunk_size = 768
    chunk_overlap = 64
    # text_splitter = CharacterTextSplitter(
    #     separator=' ',chunk_size=chunk_size, chunk_overlap=chunk_overlap
    # )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n", " ", ""]
    )
    # Split
    docs = text_splitter.split_text(text)
    for doc in docs:
        # print(doc.page_content)
        # print('++++++++++++++++++++++++++++++++++')
        results.append(doc)

    embeddings_1 = emb_model.encode([query], 
                                batch_size=12, 
                                max_length=1024, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                )['dense_vecs']
    embeddings_2 = emb_model.encode(results)['dense_vecs']
    similarity = embeddings_1 @ embeddings_2.T

    # 找出前10大值对应的索引
    top10_indices = sorted(range(len(similarity[0])), key=lambda i: similarity[0][i], reverse=True)[:10]
    emb_top_10_results = [results[i] for i in top10_indices]

    if len(emb_top_10_results) < 3:
        return emb_top_10_results
    
    reranker_list = []
    for i in emb_top_10_results:
        reranker_list.append([query,i])
    scores = reranker_model.compute_score(reranker_list, normalize=True)

    top3_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]

    final_top_3_results = [emb_top_10_results[i] for i in top3_indices]
    return final_top_3_results

def task7_rag(md_path, query):
    with open(md_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 寻找 # References 标题的位置
    pattern = r'(?i)^# References\b'  # (?i) 忽略大小写匹配
    # 使用正则表达式查找 References 部分
    match = re.search(pattern, content, re.MULTILINE)
    if match:
        # 切分并保留 # References 之前的内容
        content_before_references = content[:match.start()]
    else:
        # 如果未找到 # References，则保留全部内容
        content_before_references = content
    # print(content_before_references)
    # print(content_before_references)
    # print('++++++++++++++++++++++++++++++++++')
    # 将文本中的img路径都删除
    pattern = "!\[.*?\]\(images/[^/]*?\.jpg\)"
    text = re.sub(pattern, '', content_before_references)

    # 将连续的多个空格替换为单空格
    text = re.sub(r' +', ' ', text)
    # print(text)
    # print('++++++++++++++++++++++++++++++++++')
    # 定义正则表达式，匹配连续15个或更多大写字母
    text = re.sub(r'[A-Z]{15,}', '', text)

    text = text.replace('\n\n', '\n')
    # print(text)
    # print('++++++++++++++++++++++++++++++++++')

    results = []
    
    # headers_to_split_on = [
    # ("#", "Header 1"),
    # ("##", "Header 2"),
    # ("###", "Header 3"),
    # ]
    # markdown_splitter = MarkdownHeaderTextSplitter(
    #     headers_to_split_on=headers_to_split_on)
    # rs = markdown_splitter.split_text(text)
    

    chunk_size = 768
    chunk_overlap = 64
    # text_splitter = CharacterTextSplitter(
    #     separator=' ',chunk_size=chunk_size, chunk_overlap=chunk_overlap
    # )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n", " ", ""]
    )
    # Split
    docs = text_splitter.split_text(text)
    for doc in docs:
        # print(doc.page_content)
        # print('++++++++++++++++++++++++++++++++++')
        results.append(doc)

    embeddings_1 = emb_model.encode([query], 
                                batch_size=12, 
                                max_length=1024, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                )['dense_vecs']
    embeddings_2 = emb_model.encode(results)['dense_vecs']
    similarity = embeddings_1 @ embeddings_2.T

    # 找出前10大值对应的索引
    top10_indices = sorted(range(len(similarity[0])), key=lambda i: similarity[0][i], reverse=True)[:10]
    emb_top_10_results = [results[i] for i in top10_indices]

    if len(emb_top_10_results) < 2:
        return emb_top_10_results
    
    reranker_list = []
    for i in emb_top_10_results:
        reranker_list.append([query,i])
    scores = reranker_model.compute_score(reranker_list, normalize=True)

    top3_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2]

    final_top_3_results = [emb_top_10_results[i] for i in top3_indices]
    return final_top_3_results



if __name__=='__main__':
    embedding_model_path = '/bohr/LLM-model-6ukg/v1/bge-m3'
    reranker_model_path = '/bohr/LLM-model-6ukg/v1/bge-reranker-v2-m3'

    # 加载embedding模型
    emb_model = BGEM3FlagModel(embedding_model_path,  use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    # 加载reranker模型
    reranker_model = FlagReranker(reranker_model_path, use_fp16=True)
    
    DATA_PATH=os.getenv('DATA_PATH')
    if not DATA_PATH:
        DATA_PATH='/bohr/exampleData-pi6b/v6'
        print("Warning: DATA_PATH environment variable is not set. Using default path:", DATA_PATH)
    test_input_path = DATA_PATH+'/question.jsonl'

    new_data = []
    with jsonlines.open(test_input_path,'r') as data:
        for sample in data:
            instruction = 'The content of the paper is as follows:\n\n'
            task = sample['task']
            base_path = 'test_mid_mds'
            question = sample['input'][-1]['content']
            # try:
            if task in '134':
                new_data.append(sample)
                # continue
            # elif task == '3':
            #     doi = sample['doi'].replace('/','_').replace(' (Supporting Information)', '_si')
            #     md_dir = f'{base_path}/task{task}/{doi}'
            #     query = question.strip().split(',')[0]
            #     content = task3_rag(f'{md_dir}/{doi}.md', query)
            #     img_path = []
            #     for text in content:
            #         match = re.search(r'images/[^)]+\.jpg', text)
            #         if match:
            #             img_path.append(match.group(0))
            #     sample['img_path'] = img_path
            #     new_data.append(sample)
            elif task == '6':
                doi = sample['doi'].replace('/','_').replace(' (Supporting Information)', '_si')
                md_dir = f'{base_path}/task{task}/{doi}'
                query = 'Table \n ' + question.replace('In the upper paper, what is the ', '')
                content = task6_rag(f'{md_dir}/new_{doi}.md', query)
                img_path = []
                for text in content:
                    match = re.search(r'images/[^)]+\.jpg', text)
                    if match:
                        img_path.append(match.group(0))
                        break
                sample['img_path'] = img_path
                new_data.append(sample)
            elif task == '2':
                doi = sample['doi'].replace('/','_').replace(' (Supporting Information)', '_si')
                md_dir = f'{base_path}/task{task}/{doi}'
                query = question.replace('In the upper paper, Is the processing heat treatment technique ', '')
                content = task2_rag(f'{md_dir}/new_{doi}.md', query)
                attach_paper_content = instruction + '\n\n'.join(content)
                sample['input'].append(
                    {
                        "role":"user",
                        "content":attach_paper_content
                    }
                )
                new_data.append(sample)
                # break
                # continue
            elif task == '5':
                doi = sample['doi'].replace('/','_').replace(' (Supporting Information)', '_si')
                md_dir = f'{base_path}/task{task}/{doi}'
                query = question.split('\n')[-1]+"|\'inputs\':" + "|\'conditions\':" + "|\'outcomes\':"
                content = task5_rag(f'{md_dir}/new_{doi}.md', query)
                attach_paper_content = instruction + '\n\n'.join(content)
                sample['input'].append(
                    {
                        "role":"user",
                        "content":attach_paper_content
                    }
                )
                new_data.append(sample)
            else:
                doi = sample['doi'].replace('/','_').replace(' (Supporting Information)', '_si')
                md_dir = f'{base_path}/task{task}/{doi}'
                content = task7_rag(f'{md_dir}/{doi}.md', question)
                attach_paper_content = instruction + '\n\n'.join(content)
                sample['input'].append(
                    {
                        "role":"user",
                        "content":attach_paper_content
                    }
                )
                new_data.append(sample)
            # except:
            #     new_data.append(sample)
    with jsonlines.open('question_rag.jsonl','w') as file:
        file.write_all(new_data)