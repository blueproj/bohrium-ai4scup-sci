from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers.generation import GenerationConfig
import torch
import jsonlines
from PIL import Image
import fitz
import os
from peft import LoraConfig,TaskType,get_peft_model,PeftModel

def vl_chat(img_path, question, doi):
    # base_path = 'test_mid_mds/task3'
    content = []
    for img in img_path:
        # print(img)
        # image = Image.open(f"{base_path}/{doi}/{img}").convert('RGB')
        image = Image.open(img).convert('RGB')
        content.append(image)
    content.append(question)
    # print(content)
    msgs = [{'role': 'user', 'content':content}]
    res = vl_model.chat(
        image=None,
        msgs=msgs,
        tokenizer=vl_tokenizer,
        sampling=False,
        temperature=0.01
    )
    print(res)
    return res

def pdf2img(pdf_path, page, doi):
    base_path = "task3_imgs"
    pdf_document = fitz.open(pdf_path)
    page = pdf_document.load_page(page-1)
    dpi = 300
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
    pix.save(f'{base_path}/{doi}.jpg')

if __name__=='__main__':
    vl_model_path = '/bohr/minicpm-bb6u/v1'
    lora_model_path = '/bohr/sci-lora-model-q9h7/v14/task3-lora-0912-2'
    # 加载vl模型
    vl_model = AutoModel.from_pretrained(vl_model_path, trust_remote_code=True, attn_implementation='sdpa', torch_dtype=torch.bfloat16)
    vl_tokenizer = AutoTokenizer.from_pretrained(vl_model_path, trust_remote_code=True)
    vl_model = PeftModel.from_pretrained(
                vl_model,
                lora_model_path,
                device_map="auto",
                trust_remote_code=True
            )
    vl_model = vl_model.eval().cuda()
    test_path = 'question_rag.jsonl'
    task3_questions = []
    with jsonlines.open(test_path,'r') as data:
        for line in data:
            if line['task']=='3':
                task3_questions.append(line)

    DATA_PATH=os.getenv('DATA_PATH')
    if not DATA_PATH:
        DATA_PATH='/bohr/exampleData-pi6b/v6'
        print("Warning: DATA_PATH environment variable is not set. Using default path:", DATA_PATH)
    pdfs = DATA_PATH+'/pdfs/'

    task3_answer = []
    for sample in task3_questions:
        question = '\n'.join([x['content'] for x in sample['input']])
        doi = sample['doi'].replace('/','_').replace(' (Supporting Information)', '_si')
        pdf_path = pdfs + f'/{doi}.pdf'
        page = sample['pages'][0]
        pdf2img(pdf_path, page, doi)
        img_path = f"task3_imgs/{doi}.jpg"
        try:
            sample['ideal'] = vl_chat([img_path], question, doi)
            # print(sample['ideal'])
        except Exception as e:
            print(e)
            import random
            print(sample)
            sample['ideal'] = random.choice(sample['option'])
        task3_answer.append(sample)

    with jsonlines.open('task3_answer.jsonl', 'w') as file:
        file.write_all(task3_answer)