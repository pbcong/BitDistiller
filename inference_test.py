from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def get_response(message, model_path):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path)

    prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate output
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(generated_text)
    return generated_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()
    model_path = args.model_path
    message = "What are you?"
    get_response(message, model_path)