from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

from config import MODEL_CONFIG, MAX_SEQ_LENGTH, PEFT_CONFIG, TRAINING_ARGS
from data_prep import get_gsm8k_questions, xmlcount_reward_func, soft_format_reward_func, strict_format_reward_func, \
    int_reward_func, correctness_reward_func

def initialize_model():
    """Initialize and return the base model and tokenizer."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_CONFIG["model_name"],
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=MODEL_CONFIG["load_in_4bit"],
        fast_inference = MODEL_CONFIG["fast_inference"],
        max_lora_rank = MODEL_CONFIG["max_lora_rank"],
        gpu_memory_utilization = MODEL_CONFIG["gpu_memory_utilization"],
        dtype=MODEL_CONFIG["dtype"],
    )
    return model, tokenizer


def setup_peft_model(model):
    """Apply PEFT configuration to the model."""
    return FastLanguageModel.get_peft_model(model, **PEFT_CONFIG)

class ModelTrainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


    def setup_trainer(self, dataset):
        training_args = GRPOConfig(
            **TRAINING_ARGS
        )

        return GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=[
                xmlcount_reward_func,
                soft_format_reward_func,
                strict_format_reward_func,
                int_reward_func,
                correctness_reward_func,
            ],
            args=training_args,
            train_dataset=dataset,
        )

def main():
    # Initialize model
    model, tokenizer = initialize_model()
    model = setup_peft_model(model)

    dataset = get_gsm8k_questions()

    # Train model
    trainer = ModelTrainer(model, tokenizer)
    trainer_instance = trainer.setup_trainer(dataset)
    trainer_instance.train()

    # Save the model locally
    model.save_lora("grpo_saved_lora")

if __name__ == "__main__":
    main()