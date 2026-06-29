import copy
import logging
import logging
import os
import torch
import transformers
from dataclasses import dataclass, field
from datasets import load_dataset
from functools import partial
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, Trainer
from typing import Dict, Optional, Sequence, List
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np
import random
from datetime import datetime

# os.environ['DS_SKIP_CUDA_CHECK']='1'

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

def logging_builder(log_dir):
    """
    创建logger
    """
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    current_time = datetime.now()
    time_str = current_time.strftime("%Y%m%d_%H%M%S")
    file_name = F"log_{time_str}.log"

    log_file_path = os.path.join(log_dir, file_name)
    file_handler = logging.FileHandler(log_file_path, mode='w')

    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="none")
    use_lora: Optional[bool] = field(default=False)
    # lora_r = Optional[int] = field(default=8)
    # lora_alpha = Optional[int] = field(default=16)
    # quantization_bit = Optional[int] = field(default=4)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    source_length: int = field(default=8192)
    target_length: int = field(default=8192)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_deepspeed: bool = field(default=False)
    lr_scheduler_type: str = field(default="cosine_with_min_lr")
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {'min_lr': 0.0001})


def get_all_datapath(dir_name: str) -> List[str]:
    all_file_list = []

    for root, dir, file_name in os.walk(dir_name):
        for temp_file in file_name:
            standard_path = f"{root}/{temp_file}"

            all_file_list.append(standard_path)

    return all_file_list


def load_dataset_from_path(
    data_path: Optional[str] = None, cache_dir: Optional[str] = "cache_data"
) -> Dataset:
    all_file_list = get_all_datapath(data_path)  # [:1]
    data_files = {"train": all_file_list}
    extension = all_file_list[0].split(".")[-1]

    logger.info("load files %d number", len(all_file_list))

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        # cache_dir=cache_dir,
    )["train"]
    return raw_datasets


IGNORE_INDEX = -100


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    ne_pad_token_id = (
        IGNORE_INDEX if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    )
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(ne_pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def build_source_text(x: str, tokenizer) -> str:
    # messages = [
    #     {
    #         "role": "system",
    #         "content": "",
    #     },
    #     {"role": "user", "content": x},
    # ]
    text = tokenizer.apply_chat_template(
        x, tokenize=False, add_generation_prompt=True
    )
    return text


def make_train_dataset(
    tokenizer: transformers.PreTrainedTokenizer,
    data_path: str,
    data_args: DataArguments,
) -> Dataset:
    logging.warning("Loading data...")

    dataset = load_dataset_from_path(
        data_path=data_path,
    )
    logging.warning("Formatting inputs...")

    def generate_sources_targets(
        examples: Dict, tokenizer: transformers.PreTrainedTokenizer
    ):
        ins_data = examples["input"]
        output = examples["ideal"]

        len_ = len(ins_data)

        sources = [
            build_source_text(ins_data[i][:data_args.source_length], tokenizer)
            for i in range(len_)
        ]
        # sources = [i[: data_args.source_length] for i in ins_data]
        targets = [
            f"{example[:data_args.target_length-1]}{tokenizer.eos_token}"
            for example in output
        ]

        input_output = preprocess(sources=sources, targets=targets, tokenizer=tokenizer)
        examples["input_ids"] = input_output["input_ids"]
        examples["labels"] = input_output["labels"]
        return examples

    generate_sources_targets_p = partial(generate_sources_targets, tokenizer=tokenizer)

    dataset = dataset.map(
        function=generate_sources_targets_p,
        batched=True,
        desc="Running tokenizer on train dataset",
        num_proc=20,
    ).shuffle()
    return dataset


def load_model_and_tokenizer(
    model_args: ModelArguments,
    training_args: TrainingArguments,
) -> tuple:
    if training_args.use_deepspeed:
        
        bnb_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
                )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            # cache_dir=training_args.cache_dir,
            # torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            quantization_config=bnb_config,
            use_cache=False
        )

        model = prepare_model_for_kbit_training(model)


    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )

    if model_args.use_lora:
        logging.warning("Loading model to Lora")



        LORA_R = 16
        LORA_ALPHA = 32
        LORA_DROPOUT = 0.05
        TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

        config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )

    return model, tokenizer

class CustomTrainerCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs and 'grad_norm' in logs and 'learning_rate' in logs and 'epoch' in logs:
                log_message = (
                    f"step: {state.global_step} - "
                    f"loss: {logs['loss']} - "
                    f"grad_norm: {logs['grad_norm']} - "
                    f"learning_rate: {logs['learning_rate']} - "
                    f"epoch: {logs['epoch']}"
                )
                logger.info(log_message)



def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logger.info(model_args)
    logger.info(data_args)
    logger.info(training_args)

    model, tokenizer = load_model_and_tokenizer(model_args, training_args)

    with training_args.main_process_first(desc="loading and tokenization"):
        train_dataset = make_train_dataset(
            tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args
        )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=IGNORE_INDEX
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        callbacks=[CustomTrainerCallback()]
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    log_dir = '../log'
    logger = logging_builder(log_dir)
    setup_seed(929)
    train()
