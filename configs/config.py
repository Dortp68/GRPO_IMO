class CFG:
    MAX_TRAIN = 1000
    MAX_TOKENS = 2048
    NUM_GENERATIONS = 4
    USE_PEFT = True
    BATCH_SIZE = 1
    MAX_STEPS = 80

    BETA = 0.04
    LR = 1.0e-5

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    splitter = "<｜Assistant｜>"

    step_count = 10
    DEBUG = False
