# AI for Science Cup - 科学文献智能解析 | 第3名（二等奖）

> **Bohrium 平台 · AI for Science with Global Scientists 竞赛**
>
> 本项目是 2024 年 AI for Science Cup 科学文献智能解析赛道的解决方案，最终获得 **第3名（二等奖）**。

## 📌 赛题简介

本竞赛要求选手设计算法，对科学文献 PDF 进行智能解析与问答。赛题包含 7 个子任务，涵盖知识问答、表格理解、图表问答、分子式识别、化学反应式识别等多种模态：

| 任务 | 模态 | 说明 |
|------|------|------|
| Task 1 | 文本（知识问答） | 多选题，涉及物理、化学、生物等学科知识 |
| Task 2 | 文本 + 表格 | 基于 PDF 内容判断热处理工艺相关的 Yes/No 问题 |
| Task 3 | 图表 | 基于 PDF 中指定页码的图表进行多选问答 |
| Task 4 | 分子式 | 基于 PDF 中分子式图片的问答（专利文档为主） |
| Task 5 | 化学反应式 | 基于 PDF 中化学反应式的问答 |
| Task 6 | 表格 | 基于 PDF 中表格图片的多选问答 |
| Task 7 | 文本 | 基于 PDF 文本内容的问答 |

## 🏆 成绩

- **排名**: 第 3 名 / 二等奖
- **平台**: [Bohrium](https://www.bohrium.com/competitions/7922759072)
- **参赛时间**: 2024年8月 - 2024年9月

## 💡 解决方案概述

本方案主要分为 **PDF 解析**、**多模态数据处理**、**模型训练与预测** 三个部分。由于三个部分使用的工具依赖有冲突，共使用了 3 套 conda 环境。

### 技术架构

```
┌──────────────────────────────────────────────────────────────────┐
│                        整体流程                                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. PDF 解析 (MinerU)                                            │
│     PDF → Markdown + 表格图片 + 图表图片                          │
│                                                                  │
│  2. 多模态数据处理                                                │
│     ├─ 表格图片 → MiniCPM-V-2.6 → Markdown 表格文本               │
│     ├─ 反应式图片 → RxnScribe → SMILES 文本                       │
│     └─ 分子式图片 → MolScribe → SMILES 文本                       │
│                                                                  │
│  3. RAG 检索增强                                                  │
│     ├─ LangChain 文本切分                                         │
│     ├─ BGE-M3 向量召回 (Top-10)                                   │
│     └─ BGE-Reranker-v2 精排 (Top-1~3)                            │
│                                                                  │
│  4. 模型推理                                                      │
│     ├─ Task 1: Yi-1.5-34B × 3 (QLoRA) → 投票                    │
│     ├─ Task 2/5/7: Yi-1.5-34B (QLoRA) + RAG                     │
│     ├─ Task 3/6: MiniCPM-V-2.6 (LoRA SFT)                        │
│     └─ Task 4: 放弃（时间限制）                                    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 各任务详细方案

#### Task 1 — 知识问答（3模型集成投票）
- 将训练数据随机分为 3 份，分别对 **Yi-1.5-34B-Chat** 进行 QLoRA 微调
- 推理时由 3 个模型分别预测，通过**多数投票**得到最终结果
- 使用 vLLM 进行高效推理，支持 bitsandbytes 4bit 量化

#### Task 2 — 表格理解（VL提取 + RAG + LLM）
- 利用 MinerU 解析 PDF，将所有表格保存为图片
- 使用 **MiniCPM-V-2.6** 将表格图片转换为 Markdown 格式文本
- 将提取的表格文本替换回 Markdown，然后进行 RAG 检索
- 使用微调后的 **Yi-1.5-34B-Chat** 进行推理

#### Task 3 — 图表问答（VL模型微调）
- 根据 PDF 页码将整页保存为高分辨率图片（300 DPI）
- 使用比赛训练集对 **MiniCPM-V-2.6** 进行 LoRA 微调
- 通过微调后的 VL 模型直接对图片进行问答

#### Task 4 — 分子式识别（放弃）
- 前期方案：MinerU 解析 → MolScribe 将分子式图片转为 SMILES → RAG → LLM 推理
- 后期因方案完善导致运行超时，出于时间限制暂时放弃

#### Task 5 — 化学反应式识别（RxnScribe + RAG + LLM）
- 利用 MinerU 解析 PDF
- 使用 **RxnScribe** 将反应式图片转换为 SMILES 文本（反应物、条件、产物）
- 将提取的反应式文本替换回 Markdown，进行 RAG 检索
- 使用微调后的 **Yi-1.5-34B-Chat** 进行推理

#### Task 6 — 表格问答（VL模型微调）
- 利用 MinerU 解析 PDF，将表格保存为图片
- 使用 **MiniCPM-V-2.6** 将表格图片转换为 Markdown 文本
- RAG 检索最相关的表格图片
- 使用微调后的 **MiniCPM-V-2.6** 模型进行推理

#### Task 7 — 文本问答（RAG + LLM）
- 利用 MinerU 解析 PDF 为 Markdown
- 进行 RAG 检索，检索出与问题相关的内容
- 使用微调后的 **Yi-1.5-34B-Chat** 进行推理

## 🔧 技术栈

| 组件 | 技术 |
|------|------|
| PDF 解析 | [MinerU](https://github.com/opendatalab/MinerU) (magic-pdf) |
| 大语言模型 | Yi-1.5-34B-Chat |
| 多模态模型 | MiniCPM-V-2.6 |
| 化学反应识别 | [RxnScribe](https://github.com/thomas0809/RxnScribe) |
| 分子式识别 | [MolScribe](https://github.com/thomas0809/MolScribe) |
| 向量检索 | BGE-M3 (FlagEmbedding) |
| 重排序 | BGE-Reranker-v2-m3 |
| 文本切分 | LangChain |
| 训练框架 | DeepSpeed ZeRO-2, PEFT (QLoRA) |
| 推理框架 | vLLM (bitsandbytes 4bit) |

## 📁 项目结构

```
├── README.md                         # 中文说明
├── README_EN.md                      # English README
├── requirements.txt                  # 合并依赖
├── .gitignore
│
├── inference/                        # 推理代码
│   ├── infer_final.ipynb             # 推理主流程 Notebook（含方案说明）
│   ├── test_pdf2md.py                # PDF → Markdown 解析（MinerU）
│   ├── test_task2_img2table.py       # Task2: 表格图片 → Markdown 文本
│   ├── test_task5_img2react.py       # Task5: 反应式图片 → SMILES
│   ├── test_task6_img2table.py       # Task6: 表格图片 → Markdown 文本
│   ├── bulid_test_data.py            # RAG 数据构建（BGE-M3 + Reranker）
│   ├── task1_infer_1.py              # Task1 推理（模型1/3）
│   ├── task1_infer_2.py              # Task1 推理（模型2/3）
│   ├── task1_infer_3.py              # Task1 推理（模型3/3）
│   ├── task2_infer.py                # Task2 推理
│   ├── task3_infer.py                # Task3 推理（MiniCPM-V）
│   ├── task5_infer.py                # Task5 推理
│   ├── task6_infer.py                # Task6 推理（MiniCPM-V）
│   ├── task7_infer.py                # Task7 推理
│   └── magic-pdf.template.json       # MinerU 配置模板
│
├── training/                         # 训练代码
│   ├── train_final.ipynb             # 训练主流程 Notebook
│   ├── train_yi1.5.py                # Yi-1.5-34B QLoRA 训练脚本
│   ├── ds_zero2_no_offload.json      # DeepSpeed ZeRO-2 配置
│   ├── train_zero2_task1_1.sh        # Task1 训练（分片1）
│   ├── train_zero2_task1_2.sh        # Task1 训练（分片2）
│   ├── train_zero2_task1_3.sh        # Task1 训练（分片3）
│   ├── train_zero2_task2.sh          # Task2 训练
│   ├── train_zero2_task5.sh          # Task5 训练
│   ├── train_zero2_task7.sh          # Task7 训练
│   ├── task3_sft/                    # MiniCPM-V SFT 代码（Task3）
│   ├── task6_sft/                    # MiniCPM-V SFT 代码（Task6）
│   └── train_data/                   # 训练数据
│       ├── train_data_task1_1/       # Task1 训练数据（分片1）
│       ├── train_data_task1_2/       # Task1 训练数据（分片2）
│       ├── train_data_task1_3/       # Task1 训练数据（分片3）
│       ├── train_data_task2/         # Task2 训练数据
│       ├── train_data_task5/         # Task5 训练数据
│       ├── train_data_task7/         # Task7 训练数据
│       ├── train_task3_all.json      # Task3 SFT 训练数据
│       ├── train_task6_all.json      # Task6 SFT 训练数据
│       └── task3_imgs/               # Task3 训练图片（PDF页截图）
│
├── requirements/                     # 各环境依赖
│   ├── pkgs_mineru.txt               # MinerU 环境
│   ├── pkgs_rxn.txt                  # RxnScribe 环境
│   └── pkgs_vllm.txt                 # vLLM 主环境
│
├── data/                              # 数据
│   ├── trainning_data/               # 原始训练数据（各任务的 JSONL 文件）
│   └── parsed_markdowns/             # MinerU 解析后的 936 篇论文 Markdown（纯文本）
│
└── examples/                          # 示例：MinerU 解析后的 Markdown（含图片）
    ├── 10.1002_adem.201700820/
    ├── 10.1016_j.ccell.2020.09.014/
    └── 10.1016_j.corsci.2019.108187/
```

## 📦 数据下载

仓库内已包含训练数据（`data/trainning_data/`）和 936 篇论文的 Markdown 文本（`data/parsed_markdowns/`）。

以下大数据文件因体积较大，通过 **GitHub Release** 提供：

| 文件 | 大小 | 说明 |
|------|------|------|
| `pdfs.zip` | ~2.7 GB | 936 篇科学文献的原始 PDF 文件 |
| `mds-936-with-images.zip` | ~1.8 GB | MinerU 解析后的完整 Markdown（含提取的图片） |

> 请在 [Releases 页面](https://github.com/blueproj/bohrium-ai4scup-sci/releases) 下载这些文件。

## 🚀 环境配置

本方案共使用 3 套 conda 环境，因为不同工具之间的依赖存在冲突。

### 1. MinerU 环境（PDF 解析）

```bash
conda create -n mineru python=3.10 -y
conda run -n mineru pip install -r requirements/pkgs_mineru.txt
```

### 2. RxnScribe 环境（化学反应式识别）

```bash
conda create -n rxn python=3.10 -y
conda run -n rxn pip install -r requirements/pkgs_rxn.txt
```

### 3. vLLM 主环境（推理 + RAG）

```bash
pip install -r requirements/pkgs_vllm.txt
pip install deepspeed  # 训练时需要
```

## 📝 使用方法

### 训练

```bash
# 1. 安装依赖
pip install -r requirements/pkgs_vllm.txt
pip install deepspeed

# 2. 运行训练（以 Task1 为例）
cd training
bash train_zero2_task1_1.sh  # 训练分片1
bash train_zero2_task1_2.sh  # 训练分片2
bash train_zero2_task1_3.sh  # 训练分片3

# Task3/Task6 使用 MiniCPM-V SFT
cd task3_sft && bash finetune_lora_no_eval.sh
```

### 推理

详细的推理流程参见 `inference/infer_final.ipynb`，主要步骤：

1. **PDF 解析**：使用 MinerU 将 PDF 解析为 Markdown
2. **多模态转换**：将表格/反应式图片转换为文本
3. **RAG 构建**：使用 BGE-M3 + Reranker 检索相关内容
4. **模型推理**：使用 vLLM 加载微调模型进行推理
5. **结果合并**：合并各任务结果并生成提交文件

## ⚠️ 注意事项

- 代码中的模型路径（如 `/bohr/...`）为比赛平台路径，实际使用时需替换为本地路径
- 训练数据全部来自于比赛所提供的数据，未使用外部数据
- Task4 因运行超时问题被放弃，其余任务均有完整实现
- `data/parsed_markdowns/` 包含全部 936 篇论文的 Markdown 纯文本；含图片的完整版请从 [Releases](https://github.com/blueproj/bohrium-ai4scup-sci/releases) 下载
- 原始 PDF 文件请从 [Releases](https://github.com/blueproj/bohrium-ai4scup-sci/releases) 下载

## 📄 许可证

本项目仅供学习和参考用途。训练数据来源于比赛平台，请勿用于商业用途。
