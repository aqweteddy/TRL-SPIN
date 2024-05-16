# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Run the KTO training script with the commands below. In general, the optimal configuration for KTO will be similar to that of DPO.

# Full training:
python examples/scripts/kto.py \
    --model_name_or_path=trl-lib/qwen1.5-1.8b-sft \
    --per_device_train_batch_size 16 \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --lr_scheduler_type=cosine \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir=kto-aligned-model \
    --warmup_ratio 0.1 \
    --report_to wandb \
    --bf16 \
    --logging_first_step

# QLoRA:
python examples/scripts/kto.py \
    --model_name_or_path=trl-lib/qwen1.5-1.8b-sft \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type=cosine \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir=kto-aligned-model-lora \
    --warmup_ratio 0.1 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --use_peft \
    --load_in_4bit \
    --lora_target_modules=all-linear \
    --lora_r=16 \
    --lora_alpha=16
"""

from dataclasses import dataclass

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import KTOConfig, KTOTrainer, ModelConfig, get_peft_config, setup_chat_format


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the KTO training script.
    """

    dataset: str = "trl-lib/kto-mix-14k"
    
    
def convert_dpo2kto_format(x: dict[str, list]):
    prompt = [{'content': x['prompt'][0], 'role': 'user'}]
    return {
        'new_prompt': [prompt, prompt],
        'completion': [x['chosen'][0], x['rejected'][0]],
        'label': [True, False],
    }


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, KTOConfig, ModelConfig))
    script_args, kto_args, model_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # If we are aligning a base model, we use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    # Load the dataset
    dataset = load_dataset(**eval(script_args.dataset))['train']
    used = set()
    def unique(x):
        if x['prompt'] not in used:
            used.add(x['prompt'])
            return True
        return False
    dataset = dataset.filter(unique)
    dataset = dataset.shuffle(seed=42).select(range(30000))
    dataset = dataset.map(convert_dpo2kto_format, batched=True, batch_size=1, num_proc=32, remove_columns=dataset.column_names, load_from_cache_file=False)
    dataset = dataset.rename_column('new_prompt', 'prompt')
    
    # Apply chat template
    def format_dataset(example):
        example["prompt"] = tokenizer.apply_chat_template(example["prompt"], tokenize=False)
        example["completion"] = tokenizer.apply_chat_template(example["completion"], tokenize=False)
        return example
    
    formatted_dataset = dataset.map(format_dataset)

    formatted_dataset = formatted_dataset.train_test_split(test_size=0.1)
    formatted_dataset = formatted_dataset.shuffle(seed=42)

    
    # Load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    model_ref = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    
    # Initialize the KTO trainer
    kto_trainer = KTOTrainer(
        model,
        model_ref,
        args=kto_args,
        train_dataset=formatted_dataset["train"],
        eval_dataset=formatted_dataset["test"],
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Train and push the model to the Hub
    kto_trainer.train()
    kto_trainer.save_model(kto_args.output_dir)
    kto_trainer.push_to_hub()