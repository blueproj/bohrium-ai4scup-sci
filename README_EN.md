# AI for Science Cup - Scientific Paper Intelligent Parsing | 3rd Place (2nd Prize)

English | [中文](README.md)

> **Bohrium Platform · AI for Science with Global Scientists Competition**
>
> This project is the solution for the 2024 AI for Science Cup - Scientific Paper Intelligent Parsing track, achieving **3rd place (2nd Prize)**.

## 📌 Competition Overview

This competition requires participants to design algorithms for intelligent parsing and question answering of scientific literature PDFs. The competition consists of 7 sub-tasks covering various modalities including knowledge QA, table understanding, figure QA, molecular formula recognition, and chemical reaction recognition:

| Task | Modality | Description |
|------|----------|-------------|
| Task 1 | Text (Knowledge QA) | Multiple choice questions on physics, chemistry, biology, etc. |
| Task 2 | Text + Table | Yes/No questions about heat treatment processes based on PDF content |
| Task 3 | Figure | Multiple choice QA based on specified PDF pages containing figures |
| Task 4 | Molecular Formula | QA based on molecular formula images in PDFs (mostly patent documents) |
| Task 5 | Chemical Reaction | QA based on chemical reaction equations in PDFs |
| Task 6 | Table | Multiple choice QA based on table images in PDFs |
| Task 7 | Text | QA based on PDF text content |

## 🏆 Results

- **Ranking**: 3rd Place / 2nd Prize
- **Platform**: [Bohrium](https://www.bohrium.com/competitions/7922759072)
- **Duration**: August 2024 - September 2024

## 💡 Solution Overview

The solution is divided into three main components: **PDF Parsing**, **Multimodal Data Processing**, and **Model Training & Inference**. Due to dependency conflicts between tools, three separate conda environments are used.

### Technical Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Overall Pipeline                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. PDF Parsing (MinerU)                                         │
│     PDF → Markdown + Table Images + Figure Images               │
│                                                                  │
│  2. Multimodal Data Processing                                   │
│     ├─ Table Images → MiniCPM-V-2.6 → Markdown Table Text       │
│     ├─ Reaction Images → RxnScribe → SMILES Text                 │
│     └─ Molecule Images → MolScribe → SMILES Text                 │
│                                                                  │
│  3. RAG Retrieval                                                │
│     ├─ LangChain Text Splitting                                  │
│     ├─ BGE-M3 Vector Recall (Top-10)                            │
│     └─ BGE-Reranker-v2 Re-ranking (Top-1~3)                     │
│                                                                  │
│  4. Model Inference                                              │
│     ├─ Task 1: Yi-1.5-34B × 3 (QLoRA) → Voting                  │
│     ├─ Task 2/5/7: Yi-1.5-34B (QLoRA) + RAG                     │
│     ├─ Task 3/6: MiniCPM-V-2.6 (LoRA SFT)                       │
│     └─ Task 4: Abandoned (time constraint)                       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Task-by-Task Approach

#### Task 1 — Knowledge QA (3-Model Ensemble Voting)
- Split training data into 3 parts, fine-tune **Yi-1.5-34B-Chat** with QLoRA separately
- During inference, 3 models predict independently; final result obtained via **majority voting**
- vLLM used for efficient inference with bitsandbytes 4bit quantization

#### Task 2 — Table Understanding (VL Extraction + RAG + LLM)
- Parse PDF with MinerU, save all tables as images
- Use **MiniCPM-V-2.6** to convert table images to Markdown text
- Replace extracted table text back into Markdown, then perform RAG retrieval
- Use fine-tuned **Yi-1.5-34B-Chat** for inference

#### Task 3 — Figure QA (VL Model Fine-tuning)
- Save the specified PDF page as a high-resolution image (300 DPI)
- Fine-tune **MiniCPM-V-2.6** using the competition training set
- Use the fine-tuned VL model to directly answer questions about the image

#### Task 4 — Molecular Formula Recognition (Abandoned)
- Initial approach: MinerU parsing → MolScribe for molecule images to SMILES → RAG → LLM inference
- Abandoned due to runtime timeout issues as the solution was refined

#### Task 5 — Chemical Reaction Recognition (RxnScribe + RAG + LLM)
- Parse PDF with MinerU
- Use **RxnScribe** to convert reaction images to SMILES text (reactants, conditions, products)
- Replace extracted reaction text back into Markdown, perform RAG retrieval
- Use fine-tuned **Yi-1.5-34B-Chat** for inference

#### Task 6 — Table QA (VL Model Fine-tuning)
- Parse PDF with MinerU, save tables as images
- Use **MiniCPM-V-2.6** to convert table images to Markdown text
- RAG to retrieve the most relevant table image
- Use fine-tuned **MiniCPM-V-2.6** model for inference

#### Task 7 — Text QA (RAG + LLM)
- Parse PDF to Markdown with MinerU
- Perform RAG retrieval for content relevant to the question
- Use fine-tuned **Yi-1.5-34B-Chat** for inference

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| PDF Parsing | [MinerU](https://github.com/opendatalab/MinerU) (magic-pdf) |
| LLM | Yi-1.5-34B-Chat |
| Vision-Language Model | MiniCPM-V-2.6 |
| Reaction Recognition | [RxnScribe](https://github.com/thomas0809/RxnScribe) |
| Molecule Recognition | [MolScribe](https://github.com/thomas0809/MolScribe) |
| Vector Retrieval | BGE-M3 (FlagEmbedding) |
| Re-ranking | BGE-Reranker-v2-m3 |
| Text Splitting | LangChain |
| Training | DeepSpeed ZeRO-2, PEFT (QLoRA) |
| Inference | vLLM (bitsandbytes 4bit) |

## 📁 Project Structure

```
├── README.md                         # Chinese README
├── README_EN.md                      # English README (this file)
├── requirements.txt                  # Combined dependencies
├── .gitignore
│
├── inference/                        # Inference code
│   ├── infer_final.ipynb             # Main inference notebook (with solution overview)
│   ├── test_pdf2md.py                # PDF → Markdown parsing (MinerU)
│   ├── test_task2_img2table.py       # Task2: Table image → Markdown text
│   ├── test_task5_img2react.py       # Task5: Reaction image → SMILES
│   ├── test_task6_img2table.py       # Task6: Table image → Markdown text
│   ├── bulid_test_data.py            # RAG data construction (BGE-M3 + Reranker)
│   ├── task1_infer_1.py              # Task1 inference (model 1/3)
│   ├── task1_infer_2.py              # Task1 inference (model 2/3)
│   ├── task1_infer_3.py              # Task1 inference (model 3/3)
│   ├── task2_infer.py                # Task2 inference
│   ├── task3_infer.py                # Task3 inference (MiniCPM-V)
│   ├── task5_infer.py                # Task5 inference
│   ├── task6_infer.py                # Task6 inference (MiniCPM-V)
│   ├── task7_infer.py                # Task7 inference
│   └── magic-pdf.template.json       # MinerU config template
│
├── training/                         # Training code
│   ├── train_final.ipynb             # Main training notebook
│   ├── train_yi1.5.py                # Yi-1.5-34B QLoRA training script
│   ├── ds_zero2_no_offload.json      # DeepSpeed ZeRO-2 config
│   ├── train_zero2_task1_1.sh        # Task1 training (split 1)
│   ├── train_zero2_task1_2.sh        # Task1 training (split 2)
│   ├── train_zero2_task1_3.sh        # Task1 training (split 3)
│   ├── train_zero2_task2.sh          # Task2 training
│   ├── train_zero2_task5.sh          # Task5 training
│   ├── train_zero2_task7.sh          # Task7 training
│   ├── task3_sft/                    # MiniCPM-V SFT code (Task3)
│   ├── task6_sft/                    # MiniCPM-V SFT code (Task6)
│   └── train_data/                   # Training data
│       ├── train_data_task1_1/       # Task1 training data (split 1)
│       ├── train_data_task1_2/       # Task1 training data (split 2)
│       ├── train_data_task1_3/       # Task1 training data (split 3)
│       ├── train_data_task2/         # Task2 training data
│       ├── train_data_task5/         # Task5 training data
│       ├── train_data_task7/         # Task7 training data
│       ├── train_task3_all.json      # Task3 SFT training data
│       ├── train_task6_all.json      # Task6 SFT training data
│       └── task3_imgs/               # Task3 training images (PDF page screenshots)
│
├── requirements/                     # Per-environment dependencies
│   ├── pkgs_mineru.txt               # MinerU environment
│   ├── pkgs_rxn.txt                  # RxnScribe environment
│   └── pkgs_vllm.txt                 # vLLM main environment
│
├── data/                              # Data
│   ├── trainning_data/               # Original training data (JSONL files for each task)
│   └── parsed_markdowns/             # MinerU-parsed Markdown for 936 papers (text only)
│
└── examples/                          # Sample: MinerU parsed Markdown (with images)
    ├── 10.1002_adem.201700820/
    ├── 10.1016_j.ccell.2020.09.014/
    └── 10.1016_j.corsci.2019.108187/
```

## 📦 Data Download

The repository already includes training data (`data/trainning_data/`) and Markdown text for all 936 papers (`data/parsed_markdowns/`).

The following large data files are available via **GitHub Release** due to size:

| File | Size | Description |
|------|------|-------------|
| `pdfs_part1.zip` | ~1.3 GB | Original PDF files of first 468 papers |
| `pdfs_part2.zip` | ~1.2 GB | Original PDF files of last 468 papers |
| `mds-936-with-images.zip` | ~1.7 GB | Complete MinerU-parsed Markdown (with extracted images) |

> Please download these files from the [Releases page](https://github.com/blueproj/bohrium-ai4scup-sci/releases).

## 🚀 Environment Setup

Three conda environments are required due to dependency conflicts between different tools.

### 1. MinerU Environment (PDF Parsing)

```bash
conda create -n mineru python=3.10 -y
conda run -n mineru pip install -r requirements/pkgs_mineru.txt
```

### 2. RxnScribe Environment (Chemical Reaction Recognition)

```bash
conda create -n rxn python=3.10 -y
conda run -n rxn pip install -r requirements/pkgs_rxn.txt
```

### 3. vLLM Main Environment (Inference + RAG)

```bash
pip install -r requirements/pkgs_vllm.txt
pip install deepspeed  # Required for training
```

## 📝 Usage

### Training

```bash
# 1. Install dependencies
pip install -r requirements/pkgs_vllm.txt
pip install deepspeed

# 2. Run training (Task1 as example)
cd training
bash train_zero2_task1_1.sh  # Train split 1
bash train_zero2_task1_2.sh  # Train split 2
bash train_zero2_task1_3.sh  # Train split 3

# Task3/Task6 use MiniCPM-V SFT
cd task3_sft && bash finetune_lora_no_eval.sh
```

### Inference

See `inference/infer_final.ipynb` for the detailed inference pipeline:

1. **PDF Parsing**: Use MinerU to parse PDFs into Markdown
2. **Multimodal Conversion**: Convert table/reaction images to text
3. **RAG Construction**: Use BGE-M3 + Reranker to retrieve relevant content
4. **Model Inference**: Use vLLM to load fine-tuned models for inference
5. **Result Merging**: Merge results from all tasks and generate submission file

## ⚠️ Notes

- Model paths in the code (e.g., `/bohr/...`) are competition platform paths and need to be replaced with local paths
- All training data comes from the competition; no external data was used
- Task4 was abandoned due to runtime timeout; all other tasks have complete implementations
- `data/parsed_markdowns/` contains Markdown text for all 936 papers; the full version with images can be downloaded from [Releases](https://github.com/blueproj/bohrium-ai4scup-sci/releases)
- Original PDF files can be downloaded from [Releases](https://github.com/blueproj/bohrium-ai4scup-sci/releases)

## 📄 License

This project is for learning and reference purposes only. Training data is sourced from the competition platform and should not be used for commercial purposes.
