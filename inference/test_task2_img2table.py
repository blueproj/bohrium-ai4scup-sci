import jsonlines
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModel
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)

import re
from functools import partial

from langchain.text_splitter import MarkdownHeaderTextSplitter,MarkdownTextSplitter
from FlagEmbedding import BGEM3FlagModel,FlagReranker
from PIL import Image
import os
import time

def task2_vl_chat(img_path, question=None):
    '''
    task2,利用vl从图片中抽取表格
    '''
    image = Image.open(img_path).convert('RGB')
    question = "If there is a table in the image, extract it in markdown format. Otherwise, return 'None'."
    msgs = [{'role': 'user', 'content': [image, question]}]
    res = vl_model.chat(
        image=None,
        msgs=msgs,
        tokenizer=vl_tokenizer,
        sampling=False,
        temperature=0.01
    )
    print(res)
    return res

def image2table_1(match, base_path):
    '''
    将markdown中的表格图片替换为表格
    '''
    alt_text = match.group(1)
    image_url = match.group(2)
    new_content = task2_vl_chat(f'{base_path}/{image_url}').replace('\n\n', '\n')
    return f'![{alt_text}]({new_content})'

def image2table_2(match, base_path):
    '''
    将markdown中的表格图片替换为表格
    '''
    alt_text = match.group(2)
    image_url = match.group(1)
    new_content = task2_vl_chat(f'{base_path}/{image_url}').replace('\n\n', '\n')
    return f'![{alt_text}]({new_content})'


def task2_process():
    base_path = 'test_mid_mds/task2'
    for subdir in os.listdir(base_path):
        print(f"---------------------{subdir}-----------------------------")
        md_dir = f"{base_path}/{subdir}"
        if os.path.exists(f'{md_dir}/new_{subdir}.md'):
            print("文件存在")
        else:
            try:
                with open(f'{md_dir}/{subdir}.md', 'r', encoding='utf-8') as file:
                    content = file.read()
                replace_image_1 = partial(image2table_1, base_path=md_dir)
                replace_image_2 = partial(image2table_2, base_path=md_dir)
                new_content1 = re.sub(r'!\[(.+?)\]\((images.*?)\)', replace_image_1, content)
                new_content2 = re.sub(r'((?:Table|TABLE).*\n!\[.*\])\((images/.*\.jpg)\)', replace_image_1, new_content1)
                new_content3 = re.sub(r'!\[.*\]\((images/.*\.jpg)\)[ \t]*\n.*((?:Table|TABLE)\s[^\n]+)', replace_image_2, new_content2)
                with open(f'{md_dir}/new_{subdir}.md', 'w', encoding='utf-8') as file:
                    file.write(new_content3)
            except Exception as e:
                print(e)
                pass

if __name__=='__main__':
    start = time.time()
    vl_model_path = '/bohr/minicpm-bb6u/v1'
        # 加载vl模型
    vl_model = AutoModel.from_pretrained(vl_model_path, trust_remote_code=True, attn_implementation='sdpa', torch_dtype=torch.bfloat16)
    vl_tokenizer = AutoTokenizer.from_pretrained(vl_model_path, trust_remote_code=True)
    vl_model = vl_model.eval().cuda()
    task2_process()
    end = time.time()
    elapsed_time = end - start
    print(f"程序运行时间: {elapsed_time} 秒")
