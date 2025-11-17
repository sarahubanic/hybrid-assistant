from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def init_model():
    print("Loading Phi-2 model...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/phi-2",
        trust_remote_code=True
    )
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=256):
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = inputs.input_ids.to(device)
    model.to(device)
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    model, tokenizer = init_model()
    
    # Test prompts
    prompts = [
        "What's the capital of France?",
        "Explain how a car engine works in 2 sentences:",
        "Write a Python function to check if a number is prime:"
    ]
    
    print("\nTesting model responses:")
    for prompt in prompts:
        print("\nPrompt:", prompt)
        response = generate_response(model, tokenizer, prompt)
        print("Response:", response)

if __name__ == "__main__":
    main()