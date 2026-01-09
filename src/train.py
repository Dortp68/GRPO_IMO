import datetime
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOConfig, GRPOTrainer
from transformers import PrinterCallback


def patch_model():
    PatchFastRL("GRPO", FastLanguageModel)


def load_model(config):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.MAX_TOKENS,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=64,
        gpu_memory_utilization=0.7,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=32,
        use_gradient_checkpointing="unsloth",
    )
    return model, tokenizer


def setup_training_args(config):
    dtstr = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_directory = f"./DEEPSEEK-GRPO-{dtstr}"

    training_args = GRPOConfig(
        output_dir=output_directory,
        use_vllm=True,
        learning_rate=config.LR,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=1,
        max_steps=config.MAX_STEPS,
        max_completion_length=config.MAX_TOKENS,
        num_generations=config.NUM_GENERATIONS,
        logging_steps=config.step_count,
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=config.step_count,
        report_to="none",
        overwrite_output_dir="True",
    )
    return training_args, output_directory


def train_model(model, training_args, dataset, reward_funcs):
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset["train"],
        callbacks=[PrinterCallback()],
    )
    trainer.train()
    return trainer
