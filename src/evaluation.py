from data_prep import get_gsm8k_questions, extract_answer_from_model_output, get_gsm8k_questions_with_no_sys_prompt
from finetune import initialize_model, setup_peft_model
from vllm import SamplingParams


def run_inference(model, tokenizer, prompt, lora_request=None):
    text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

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


def clean_numbers(string_with_formatted_number: str):
    return string_with_formatted_number.replace('$', '') \
        .replace(',', '') \
        .replace('.', '') \
        .strip()


def evaluate(model, tokenizer, dataset, num_of_answers, lora_request=None):
    num_correct_answers = 0

    for idx, data in enumerate(dataset, start=1):
        if idx % (num_of_answers + 1) == 0:
            break
        prompt = data["prompt"]
        question = data["question"]
        answer = data["answer"]
        output = run_inference(model, tokenizer, prompt, lora_request)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Model answer: {output}")

        if answer in clean_numbers(extract_answer_from_model_output(output)):
            num_correct_answers = num_correct_answers + 1

    return num_correct_answers


def main():
    num_of_answers = 100
    model, tokenizer = initialize_model()
    model = setup_peft_model(model)

    dataset = get_gsm8k_questions_with_no_sys_prompt(split="test")
    num_correct_answers = evaluate(model, tokenizer, dataset, num_of_answers)

    dataset = get_gsm8k_questions(split="test")
    lora_request = model.load_lora("../grpo_saved_lora")
    num_correct_answers_grpo = evaluate(model, tokenizer, dataset, num_of_answers, lora_request)

    print(
        f"Original model - Correctness: {num_correct_answers / num_of_answers} ({num_correct_answers}/{num_of_answers})")
    print(
        f"GRPO model - Correctness: {num_correct_answers_grpo / num_of_answers} ({num_correct_answers_grpo}/{num_of_answers})")


if __name__ == "__main__":
    main()
