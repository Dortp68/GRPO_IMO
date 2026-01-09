import torch
from tqdm import tqdm
import numpy as np


def gen(model, tokenizer, text, max_tokens, splitter):
    model_input = tokenizer(text, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        tok = model.generate(
            **model_input,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.pad_token_type_id,
        )
        outputs = []
        for i in range(len(tok)):
            res = tokenizer.decode(tok[i], skip_special_tokens=True)
            output = res.split(splitter)[-1]
            outputs.append(output)
        return outputs[0] if len(outputs) == 1 else outputs


def evaluate_rewards(
    model, tokenizer, dataset, reward_functions, max_tokens, num_generations, splitter
):
    completions = []
    other_info = []
    for example in tqdm(dataset):
        txt = example["prompt"]
        kw = {k: v for k, v in example.items() if k not in {"prompt", "completion"}}
        for _ in range(num_generations):
            other_info.append(kw)

        completion = gen(
            model, tokenizer, [txt] * num_generations, max_tokens, splitter
        )
        if isinstance(completion, str):
            completions.append(completion)
        else:
            completions += completion

    kwargs = {k: [d[k] for d in other_info] for k in other_info[0].keys()}
    res = {}
    for nm, reward_func in reward_functions.items():
        v = reward_func(completions=completions, **kwargs)
        print(nm, np.mean(v))
        res[nm] = np.mean(v)
    return res
