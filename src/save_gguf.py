import os

from unsloth import FastLanguageModel
from config import MAX_SEQ_LENGTH


def main():
    # Initialize model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="../grpo_saved_lora",
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    # Save the model locally
    # model.save_pretrained_gguf("Qwen2.5-3B-Instruct-Math-Reasoning-GGUF", tokenizer, quantization_method=["q5_k_m"])

    # Push to hub
    model.push_to_hub_gguf(
        "ugriffo/Qwen2.5-3B-Instruct-Math-Reasoning-GGUF",
        tokenizer,
        quantization_method=["f16", "q8_0", "q4_k_m" , "q5_k_m"],
        token=os.getenv("HUGGINGFACE_TOKEN"),
    )

if __name__ == "__main__":
    main()