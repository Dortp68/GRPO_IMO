import re
import pandas as pd
from datasets import Dataset


def extract_boxed_text(text):
    pattern = r"oxed{(.*?)}"
    matches = re.findall(pattern, text)
    if not matches:
        return ""
    for match in matches[::-1]:
        if match != "":
            return match
    return ""


def is_valid_answer(s):
    try:
        if float(s) == int(s):
            i = int(s)
            return 0 <= i < 1000
        else:
            return False
    except ValueError:
        return False


def prepare_dataset(path, max_train, config):
    df = pd.read_parquet(path)
    df = df.reset_index().rename({"index": "id"}, axis=1)
    df["answer"] = df["solution"].map(extract_boxed_text)

    mask = df["answer"].map(is_valid_answer)
    df = df[mask]

    df = df.iloc[:max_train]
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.1)
    return dataset


def create_prompt(sample, tokenizer, splitter):
    question = sample["problem"]
    chat = [
        {
            "role": "system",
            "content": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>",
        },
        {
            "role": "user",
            "content": question
            + " Return final answer within \\boxed{}, after taking modulo 1000.",
        },
    ]
    sample["prompt"] = tokenizer.apply_chat_template(
        conversation=chat, tokenize=False, add_generation_prompt=True
    )
    return sample
