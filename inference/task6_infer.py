from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers.generation import GenerationConfig
import torch
import jsonlines
from PIL import Image
from peft import LoraConfig,TaskType,get_peft_model,PeftModel


def vl_chat(img_path, question, doi):
    base_path = 'test_mid_mds/task6'
    content = []
    for img in img_path:
        # print(img)
        image = Image.open(f"{base_path}/{doi}/{img}").convert('RGB')
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

if __name__=='__main__':
    vl_model_path = '/bohr/minicpm-bb6u/v1'
    lora_model_path = '/bohr/sci-lora-model-q9h7/v14/task6-lora-0912-2'
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
    task6_questions = []
    with jsonlines.open(test_path,'r') as data:
        for line in data:
            if line['task']=='6':
                task6_questions.append(line)

    task6_answer = []
    for sample in task6_questions:
        img_path = sample['img_path']
        question = '\n'.join([x['content'] for x in sample['input']])
        doi = sample['doi'].replace('/','_').replace(' (Supporting Information)', '_si')
        try:
            sample['ideal'] = vl_chat(img_path, question, doi)
            print(sample['ideal'])
        except Exception as e:
            print(e)
            import random
            print(sample)
            sample['ideal'] = random.choice(sample['option'])
        task6_answer.append(sample)

    with jsonlines.open('task6_answer.jsonl', 'w') as file:
        file.write_all(task6_answer)