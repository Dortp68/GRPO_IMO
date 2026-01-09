import re
import numpy as np
from Levenshtein import ratio as levenshtein_ratio


def extract_boxed_text(text):
    pattern = r"oxed{(.*?)}"
    matches = re.findall(pattern, text)
    if not matches:
        return ""
    for match in matches[::-1]:
        if match != "":
            return match
    return ""


def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>.*?oxed{(.*?)}.*?$"
    matches = [re.match(pattern, content, re.DOTALL) for content in completions]
    return [1.0 if match else 0.0 for match in matches]


def accuracy_reward_func(completions, answer, **kwargs):
    # Regular expression to capture content inside \boxed{}
    contents = [extract_boxed_text(completion) for completion in completions]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    return [1.0 if c == str(gt) else 0.0 for c, gt in zip(contents, answer)]


def levenshtein_reward_func(completions, solution, **kwargs):
    res = []
    for completion, sol in zip(completions, solution):
        if "</think>" in completion:
            t = completion.split("</think>")[-1]
            res.append(levenshtein_ratio(t, sol))
        else:
            res.append(0.0)
    return res
