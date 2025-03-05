import os
import argparse


import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model


from srcs.datamodule import JSONLDataset, DataCollatorForImageTextToText


# REF: https://github.com/roboflow/maestro

DIR = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--local_rank", type=str)
    args = parser.parse_args()

    # model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

    # https://github.com/huggingface/transformers/issues/29266
    # bnb is not compatible with deepspeed
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_type=torch.bfloat16,
    # )

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        # device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # processor는
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        use_fast=True,
    )
    processor.tokenizer.padding_side = "left"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds_config = {
        # "train_batch_size": 8,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            # "allgather_partitions": True,
            # "reduce_scatter": True,
            # "contiguous_gradients": True,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "offload_param": {"device": "cpu", "pin_memory": True},
        },
    }

    train_path = os.path.join(DIR, "dataset", "hwp", "train")
    train_anot_path = os.path.join(train_path, "annotations.jsonl")
    train_dataset = JSONLDataset(
        annotations_path=train_anot_path, images_directory_path=train_path
    )

    valid_path = os.path.join(DIR, "dataset", "hwp", "valid")
    valid_anot_path = os.path.join(valid_path, "annotations.jsonl")
    valid_dataset = JSONLDataset(
        annotations_path=valid_anot_path, images_directory_path=valid_path
    )

    system_message = """
당신은 PDF이미지의 구조화된 텍스트를 HTML로 생성하는 Vision Language Model이다.
HTML 텍스트만 생성한다. 추가적인 설명은 피한다.
"""

    data_collator = DataCollatorForImageTextToText(
        processor=processor, system_message=system_message
    )

    training_args = TrainingArguments(
        output_dir="./results",
        # gradient_accumulation_steps=2,
        per_device_train_batch_size=1,
        num_train_epochs=3,
        fp16=True,
        deepspeed=ds_config,
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
    )

    trainer.train()
