import json
import jsonlines
import os

import sys
sys.path.insert(0, '/bohr/sci-mineru-rxn-v894/v1/MinerU')

from loguru import logger
from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter

import magic_pdf.model as model_config 
model_config.__use_inside_model__ = True

from PyPDF2 import PdfFileWriter, PdfReader, PdfWriter
import time

def extrct_pdf(pdf_path, doi, task):
    '''
    利用mineru将pdf转markdown
    '''
    current_script_dir = f'{md_dir}/task{task}/{doi}'
    if not os.path.exists(current_script_dir):
        os.makedirs(current_script_dir) 
    else:
        return '0'
    pdf_bytes = open(pdf_path, "rb").read()
    model_json = []  # model_json传空list使用内置模型解析
    jso_useful_key = {"_pdf_type": "", "model_list": model_json}
    local_image_dir = os.path.join(current_script_dir, 'images')
    image_dir = str(os.path.basename(local_image_dir))
    image_writer = DiskReaderWriter(local_image_dir)
    pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
    pipe.pipe_classify()
    """如果没有传入有效的模型数据，则使用内置model解析"""
    if len(model_json) == 0:
        if model_config.__use_inside_model__:
            pipe.pipe_analyze()
        else:
            logger.error("need model list input")
            exit(1)
    pipe.pipe_parse()
    md_content = pipe.pipe_mk_markdown(image_dir, drop_mode="none")
    with open(f"{current_script_dir}/{doi}.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    return current_script_dir

# def task3_sub_pdf(pdf_path, sub_pdf_path, start_page, end_page):
#     '''
#     根据pages抽取指定页的pdf
#     '''
#     pdf_file = PdfReader(open(pdf_path, 'rb'))
#     out_file = PdfWriter()
#     for i in range(start_page, end_page):
#         out_file.add_page(pdf_file.pages[i])
    
#     outputStream = open(sub_pdf_path, 'wb')
#     out_file.write(outputStream)


if __name__=='__main__':
    mds_936 = "/bohr/mds-936-fzsd/v1/mid_mds_936"
    mds_936_list = os.listdir(mds_936)

    DATA_PATH=os.getenv('DATA_PATH')
    if not DATA_PATH:
        DATA_PATH='/bohr/exampleData-pi6b/v6'
        print("Warning: DATA_PATH environment variable is not set. Using default path:", DATA_PATH)
    pdfs = DATA_PATH+'/pdfs/'
    md_dir = 'test_mid_mds'
    # mid_pdfs = '../test_mid_pdfs'
    start_time = time.time()
    error_list = []

    test_input_path = DATA_PATH+'/question.jsonl'
    with jsonlines.open(test_input_path, 'r') as data:
        for sample in data:
            task = sample['task']
            if task in '134':
                continue
            else:
                # if task!='7':
                #     continue
                doi = sample['doi'].replace('/','_').replace(' (Supporting Information)', '_si')

                if doi in mds_936_list:
                    source_file = f"{mds_936}/{doi}"
                    destination_path = f'{md_dir}/task{task}/{doi}'
                    return_code = os.system(f'mkdir -p {destination_path}')
                    command = f'cp -rf {source_file}/* {destination_path}/'
                    return_code = os.system(command)

                pdf_path = pdfs + f'/{doi}.pdf'
                print(f"task{task} ------------- {doi}")
                try:
                    if os.path.exists(f'{md_dir}/task{task}/{doi}'):
                            print('dir exist')
                            continue
                    else:
                        _ = extrct_pdf(pdf_path, doi, task)
                    # if sample["pages"] != [1,-1]:
                    #     start_page, end_page = sample['pages']
                    #     sub_pdf_path = f'{mid_pdfs}/{doi}.pdf'
                    #     task3_sub_pdf(pdf_path, sub_pdf_path, start_page-1, end_page)
                    #     if os.path.exists(f'{md_dir}/task{task}/{doi}'):
                    #         print('dir exist')
                    #         continue
                    #     else:
                    #         _ = extrct_pdf(sub_pdf_path, doi, task)
                    # else:
                    #     if os.path.exists(f'{md_dir}/task{task}/{doi}'):
                    #         print('dir exist')
                    #         continue
                    #     else:
                    #         _ = extrct_pdf(pdf_path, doi, task)
                except:
                    error_list.append(f"{task}_{doi}")
                # break
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(error_list)
        print(f"程序运行时间: {elapsed_time} 秒")