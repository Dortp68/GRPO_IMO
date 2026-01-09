import kagglehub
from peft import PeftModel
from configs.config import CFG
from src.data_prep import prepare_dataset, create_prompt
from src.rewards import (
    format_reward_func,
    accuracy_reward_func,
    levenshtein_reward_func,
)
from src.train import patch_model, load_model, setup_training_args, train_model
from src.evaluate import evaluate_rewards


def main():
    # Download dataset
    path = kagglehub.dataset_download("artemgoncarov/math-problems-imo")
    print("Path to dataset files:", path)

    # Prepare data
    dataset = prepare_dataset(path, CFG.MAX_TRAIN, CFG)

    # Patch and load model
    patch_model()
    model, tokenizer = load_model(CFG)

    dataset = dataset.map(lambda x: create_prompt(x, tokenizer, CFG.splitter))

    # Reward functions
    reward_functions = {
        "formatting": format_reward_func,
        "accuracy": accuracy_reward_func,
        "solution_quality": levenshtein_reward_func,
    }

    # Evaluate before training
    if not CFG.DEBUG:
        original_rewards = evaluate_rewards(
            model,
            tokenizer,
            dataset["test"],
            reward_functions,
            CFG.MAX_TOKENS,
            CFG.NUM_GENERATIONS,
            CFG.splitter,
        )

    # Setup training
    training_args, output_directory = setup_training_args(CFG)

    # Train
    trainer = train_model(
        model, training_args, dataset, list(reward_functions.values())
    )

    # Load trained model
    if CFG.USE_PEFT:
        CHKPT = CFG.MAX_STEPS
        adapter_model_name = f"{output_directory}/checkpoint-{CHKPT}/"
        new_model = PeftModel.from_pretrained(model, adapter_model_name)
    else:
        new_model = model

    # Evaluate after training
    rewards = evaluate_rewards(
        new_model,
        tokenizer,
        dataset["test"],
        reward_functions,
        CFG.MAX_TOKENS,
        CFG.NUM_GENERATIONS,
        CFG.splitter,
    )
    print(rewards)


if __name__ == "__main__":
    main()
