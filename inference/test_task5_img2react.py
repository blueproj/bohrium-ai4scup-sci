import sys
sys.path.append('/bohr/sci-mineru-rxn-v894/v1/RxnScribe')
import torch
from rxnscribe import RxnScribe
import re
import os
from functools import partial
import time

def image2react(match, base_path):
    '''
    将markdown中的图片替换为反应式
    '''
    alt_text = match.group(1)
    image_url = match.group(2)
    
    predictions = model.predict_image_file(f'{base_path}/{image_url}', molscribe=True, ocr=True)
    # try:
    reacts = []
    for i in predictions:
        inputs = []
        try:
            for j in i['reactants']:
                if len(j['smiles'])<=60 and j['smiles']!='<invalid>':
                    inputs.append(j['smiles'])
        except:
            pass
        conditions = []
        try:
            for j in i['conditions']:
                 conditions.append(j['text'])
        except:
            pass
        outcomes =[]
        try:
            for j in i['products']:
                if len(j['smiles'])<=60 and j['smiles']!='<invalid>':
                    outcomes.append(j['smiles'])
        except:
            pass
        reacts.append(
            {
                'inputs': inputs,
                'conditions':conditions,
                'outcomes':outcomes
            }
        )
    return f'![{alt_text}]({reacts})'


def task5_process():
    base_path = 'test_mid_mds/task5'
    for subdir in os.listdir(base_path):
        print(f"---------------------{subdir}-----------------------------")
        md_dir = f"{base_path}/{subdir}"
        if os.path.exists(f'{md_dir}/new_{subdir}.md'):
            print("文件存在")
        else:
            # try:
            with open(f'{md_dir}/{subdir}.md', 'r', encoding='utf-8') as file:
                content = file.read()
            replace_image = partial(image2react, base_path=md_dir)
            new_content = re.sub(r'!\[(.*?)\]\((images.*?)\)', replace_image, content)
            with open(f'{md_dir}/new_{subdir}.md', 'w', encoding='utf-8') as file:
                file.write(new_content)
            # except:
                # with open(f'{md_dir}/{subdir}.md', 'r', encoding='utf-8') as file:
                #     content = file.read()
                # with open(f'{md_dir}/new_{subdir}.md', 'w', encoding='utf-8') as file:
                #     file.write(content)
        # break

if __name__=="__main__":
    start = time.time()
    ckpt_path = '/bohr/other-model-lm85/v1/RxnScribe/pix2seq_reaction_full.ckpt'
    model = RxnScribe(ckpt_path, device=torch.device('cuda'))
    task5_process()
    end = time.time()
    elapsed_time = end - start
    print(f"程序运行时间: {elapsed_time} 秒")