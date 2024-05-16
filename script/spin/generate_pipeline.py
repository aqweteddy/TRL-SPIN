from transformers import TextGenerationPipeline, AutoModelForCausalLM
from fire import Fire
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from datasets import load_dataset
import torch, random


# generate {'prompt': [multi turn], 'chosen': last turn chosen resp., 'rejected': 'last turn rejected resp.']} for each item.


def apply_chat_template_to_batch(x: list[dict], tokenizer: PreTrainedTokenizerBase):
    # with_template = tokenizer.apply_chat_template(x['messages'], add_generation_prompt=True, tokenize=False)
    messages: list[dict[str, str]] = x['messages'][0]  # batched
    # remove system messages
    if messages[0]['role'] == 'system':
        messages = messages[1:]

    prompts, chosens = [], []

    for i in range(1, len(messages), 2):
        mes = messages[:i+1]
        with_template = tokenizer.apply_chat_template(
            mes[:-1],
            add_generation_prompt=True,
            tokenize=False
        )
        prompts.append(with_template)
        chosens.append(mes[-1]['content'])
    prs = random.choices(list(zip(prompts, chosens)), k=min(1, len(prompts)))
    prompt, chosen = [p[0] for p in prs], [p[1] for p in prs]
    return {
        'prompt': prompt,
        'chosen': chosen,
        'length': [len(p) for p in prompt]
    }

BATCH_SIZE = 4
def generate(x):
    global BATCH_SIZE
    while 1:
        try:
            result = PIPE(x['prompt'],
                          max_new_tokens=2048,
                          return_full_text=False,
                          batch_size=BATCH_SIZE
                          )
            break
        except RuntimeError as e:
            BATCH_SIZE //= 2
            BATCH_SIZE = max(BATCH_SIZE, 2)
            print('retrying...')

    return {'rejected': [r['generated_text'] for r in result]}


PIPE = None

torch.inference_mode()


def main(
    dataset_kwargs: str,
    model_path: str = '/home/u3844240/checkpoints/cp/llama3-8b_cp-p1_tv-llama3-emb_ft-b8.3patch1e1',
    output_path: str = 'output_data/llama3-8b_cp-p1_tv-llama3-emb_ft-b8.3patch1e1/round0.jsonl',
    num_proc: int = 1,
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ds = load_dataset(**dataset_kwargs)
    ds = ds.shuffle().select(range(8000))

    print('num examples:', len(ds))

    ds = ds.map(lambda x: apply_chat_template_to_batch(x, tokenizer),
                num_proc=num_proc,
                batched=True,
                remove_columns=ds.column_names,
                batch_size=1)
    ds = ds.sort('length', reverse=True)
    print('after apply_chat_template_to_batch:', len(ds))
    global PIPE
    tokenizer.padding_side = 'left'
    eos_token_id = [tokenizer.convert_tokens_to_ids(
        '<|eot_id|>'), tokenizer.eos_token_id]
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map=0,
        attn_implementation='flash_attention_2',
        trust_remote_code=False)
    model.eval()
    PIPE = TextGenerationPipeline(model=model,  tokenizer=tokenizer,
                                  pad_token_id=tokenizer.pad_token_id,
                                  eos_token_id=eos_token_id)
    ds = ds.map(generate, batch_size=32, )
    ds.to_json(output_path, orient='records', lines=True, force_ascii=False)


if __name__ == '__main__':
    Fire(main)
