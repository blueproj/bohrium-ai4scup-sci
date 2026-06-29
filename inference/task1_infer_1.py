import os
import json
import jsonlines
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import torch
import numpy as np
import random
from transformers import AutoTokenizer

def template_prompt(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

if __name__=='__main__':
    tokenizer = AutoTokenizer.from_pretrained("/bohr/blue-tvh5/v3")
    test_path = 'question_rag.jsonl'
    task1_questions = []
    with jsonlines.open(test_path,'r') as data:
        for line in data:
            if line['task']=='1':
                task1_questions.append(line)
    prompts = []
    for question in task1_questions:
        prompts.append(template_prompt(tokenizer,question['input']))
    sampling_params = SamplingParams(temperature=0.01, top_p=0.7, max_tokens=256)
    lora_path = '/bohr/sci-lora-model-q9h7/v14/yi-lora-task1-1-0902'
    llm = LLM(model="/bohr/blue-tvh5/v3", dtype="bfloat16",  max_model_len=4096, enable_lora=True, enforce_eager=True, gpu_memory_utilization=0.9, quantization="bitsandbytes", load_format="bitsandbytes")
    outputs = llm.generate(prompts, sampling_params, lora_request=LoRARequest("adapter", 1, lora_path)) # 

    ans = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        ans.append(generated_text)

    task1_answer = []
    for i, obj in enumerate(task1_questions):
        obj['ideal'] = ans[i]
        task1_answer.append(obj)
    with jsonlines.open('task1_answer_1.jsonl', 'w') as file:
        file.write_all(task1_answer)