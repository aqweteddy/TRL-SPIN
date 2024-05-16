from vllm import LLM, SamplingParams
import random
from fire import Fire
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from datasets import load_dataset
import torch
# generate {'prompt': [multi turn], 'chosen': last turn chosen resp., 'rejected': 'last turn rejected resp.']} for each item.

ENGINE = None


def apply_chat_template_to_batch(x: list[dict], tokenizer: PreTrainedTokenizerBase, messages_col='messages'):
    # with_template = tokenizer.apply_chat_template(x['messages'], add_generation_prompt=True, tokenize=False)
    messages: list[dict[str, str]] = x[messages_col][0]  # batched
    # remove system messages
    if messages[0]['role'] == 'system':
        messages = messages[1:]
    
    prompts, chosens, prompt_templates = [], [], []

    for i in range(1, len(messages), 2):
        mes = messages[:i+1]
        with_template = tokenizer.apply_chat_template(
            mes[:-1],
            add_generation_prompt=True,
            tokenize=False
        )
        prompt_templates.append(with_template)
        chosens.append(mes)
        prompts.append(messages[0]['content'])
        
    return {
        'prompt': prompts,
        'chosen': chosens,
        'prompt_template': prompt_templates,
        'length': [len(p) for p in prompts]
    }

def generate_response(x: dict[str, str]):
    """batch generate response for the given prompt

    Args:
        x (list[dict[str, str]]): _description_
    """
    params = SamplingParams(
        max_tokens=4096,
    )
    rejects = ENGINE.generate(x['prompt_template'], 
                    sampling_params=params,
                    use_tqdm=False
                    )
    rejects = [{'role': 'assistant', 'content': r.outputs[0].text} for r in rejects]
    
    
    return {'rejected': [cs[:-1] + [r] for r, cs in  zip(rejects, x['chosen'])]}

def main(
    dataset_kwargs: str,
    model_path: str = '/home/u3844240/checkpoints/cp/llama3-8b_cp-p1_tv-llama3-emb_ft-b8.3patch1e1',
    output_path: str = 'output_data/llama3-8b_cp-p1_tv-llama3-emb_ft-b8.3patch1e1/round0.jsonl',
    messages_col: str = 'chosen',
    num_proc: int = 32,
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ds = load_dataset(**dataset_kwargs)
    ds = ds.shuffle().select(range(10000))

    print('num examples:', len(ds))
    
    ds = ds.map(lambda x: apply_chat_template_to_batch(x, tokenizer, messages_col=messages_col),
                num_proc=num_proc,
                batched=True,
                remove_columns=ds.column_names,
                batch_size=1)
    print('after apply_chat_template_to_batch:', len(ds))
    global ENGINE
    ENGINE = LLM(model_path,
                 dtype='b786float16',
                 gpu_memory_utilization=0.70,
                 tensor_parallel_size=1,
                 enforce_eager=True
                 )
    ds = ds.map(generate_response, batched=True, batch_size=32)
    ds = ds.remove_columns(['prompt_template'])
    ds.to_json(output_path, orient='records', lines=True, force_ascii=False)


if __name__ == '__main__':
    Fire(main)
