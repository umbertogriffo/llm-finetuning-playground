MAX_SEQ_LENGTH = 2048
MAX_PROMPT_LENGTH = 2048
LORA_RANK = 64 # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128 - Larger rank = smarter, but slower

MODEL_CONFIG = {
    "model_name": "Qwen/Qwen2.5-3B-Instruct",
    "load_in_4bit": True, # False for LoRA 16bit
    "fast_inference": True, # Enable vLLM fast inference
    "max_seq_length": MAX_SEQ_LENGTH,
    "max_lora_rank": LORA_RANK,
    "gpu_memory_utilization": 0.6,  # Reduce if out of memory
    "dtype": None,
}

PEFT_CONFIG = {
    "r": LORA_RANK,
    "lora_alpha": LORA_RANK,
    "lora_dropout": 0,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "up_proj",
        "down_proj",
        "o_proj",
        "gate_proj",
    ], # Remove QKVO if out of memory
    "use_rslora": False,
    "use_gradient_checkpointing": "unsloth", # Enable long context finetuning
    "random_state": 3407
}

TRAINING_ARGS = {
    "use_vllm": True,
    "learning_rate": 5e-6,
    "adam_beta1": 0.9,
    "adam_beta2": 0.99,
    "weight_decay": 0.1,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "optim": "paged_adamw_8bit",
    "logging_steps": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,  # Increase to 4 for smoother training
    "num_generations": 8,  # Decrease if out of memory
    "max_prompt_length": MAX_PROMPT_LENGTH,
    "max_completion_length": MAX_SEQ_LENGTH,
    "num_train_epochs": 1, # Set to 1 for a full training run
    "max_steps": 250,
    "save_steps": 250,
    "max_grad_norm": 0.1,
    "report_to": "none",  # Can use Weights & Biases
    "output_dir": "../outputs",
}