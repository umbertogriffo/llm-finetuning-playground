from data_prep import SYSTEM_PROMPT
from finetune import initialize_model, setup_peft_model
from vllm import SamplingParams

def run_inference(model, tokenizer, system_prompt=None, lora_request=None):
    if system_prompt is None:
        conversation = [
        {"role": "user", "content": "How many r's are in strawberry?"},
    ]
    else:
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "How many r's are in strawberry?"},
        ]

    text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=1024,
    )
    output = model.fast_generate(
        [text],
        sampling_params=sampling_params,
        lora_request=lora_request,
    )[0].outputs[0].text

    return output


def main():

    model, tokenizer = initialize_model()
    model = setup_peft_model(model)

    output = run_inference(model, tokenizer)
    print(f"Original model answer: {output}")

    lora_request = model.load_lora("../grpo_saved_lora")
    output = run_inference(model, tokenizer, SYSTEM_PROMPT, lora_request)
    print(f"GRPO model answer: {output}")

if __name__ == "__main__":
    main()