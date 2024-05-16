from datasets import load_dataset
import copy


def convert(x: dict):
    try:
        x['rejected'] = x['rejected'].replace(
            '<|start_header_id|>assistant<|end_header_id|>', '')
    except:
        print(x)
    prompts = x['prompt'].split('<|eot_id|>')[:-1]
    assert len(prompts) % 2 == 1
    for i in range(0, len(prompts), 2):
        prompts[i] = prompts[i].replace('<|start_header_id|>user<|end_header_id|>', '').replace(
            '<|start_header_id|>assistant<|end_header_id|>', '').replace('<|begin_of_text|>', '').strip()
        prompts[i] = {'role': 'user', 'content': prompts[i]}
        if i + 1 < len(prompts):
            prompts[i+1] = prompts[i+1].replace('<|start_header_id|>user<|end_header_id|>', '').replace(
                '<|start_header_id|>assistant<|end_header_id|>', '').replace('<|begin_of_text|>', '').strip()
            prompts[i+1] = {'role': 'assistant', 'content': prompts[i+1]}

    x['prompt'] = prompts[0]['content']
    x['chosen'] = prompts + [{'role': 'assistant', 'content': x['chosen']}]
    x['rejected'] = prompts + [{'role': 'assistant', 'content': x['rejected']}]
    assert isinstance(x['chosen'], list)
    assert isinstance(x['rejected'], list)
    assert isinstance(x['prompt'], str)
    return {
        'prompt': x['prompt'],
        'chosen': x['chosen'],
        'rejected': x['rejected']
    }


def fix(
    x: dict
):
    chosen = x['chosen'].split('<|eot_id|>')[:-1]

    for i in range(0, len(chosen), 2):
        chosen[i] = chosen[i].replace('<|start_header_id|>user<|end_header_id|>', '').replace(
            '<|start_header_id|>assistant<|end_header_id|>', '').replace('<|begin_of_text|>', '').strip()
        chosen[i] = {'role': 'user', 'content': chosen[i]}
        if i + 1 < len(chosen):
            chosen[i+1] = chosen[i+1].replace('<|start_header_id|>user<|end_header_id|>', '').replace(
                '<|start_header_id|>assistant<|end_header_id|>', '').replace('<|begin_of_text|>', '').strip()
            chosen[i+1] = {'role': 'assistant', 'content': chosen[i+1]}
    rejected = chosen[:-1] + [{'role': 'assistant', 'content': x['rejected']}]
    return {
        'prompt': x['prompt'],
        'chosen': chosen,
        'rejected': rejected
    }


def main(
    data_kwargs: dict,
    output_path: str,
):
    ds = load_dataset(**data_kwargs)['train']
    # ds = ds.filter(lambda x: not isinstance(x['rejected'], str), num_proc=32)
    # ds = ds.map(convert, remove_columns=ds.column_names, num_proc=16)
    ds = ds.map(fix, remove_columns=ds.column_names, num_proc=32)
    ds = ds.shuffle()
    ds.to_json(output_path, force_ascii=False, orient='records', lines=True)


if __name__ == '__main__':
    from fire import Fire
    Fire(main)
