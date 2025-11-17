import os
from llama_cpp import Llama

def test_load_model():
    # Get the absolute path to the model
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_path, "models", "Llama-3.2-3B-Instruct-uncensored.Q8_0.gguf")
    
    print(f"Testing model loading from: {model_path}")
    print(f"File exists: {os.path.exists(model_path)}")
    print(f"File size: {os.path.getsize(model_path) / (1024*1024*1024):.2f} GB")
    
    try:
        print("\nAttempting to load the model...")
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            verbose=True
        )
        print("\nModel loaded successfully!")
        
        # Try a simple inference
        print("\nTesting inference...")
        output = llm("Hello, are you working?", max_tokens=20)
        print("\nModel response:", output['choices'][0]['text'])
        
    except Exception as e:
        print(f"\nError loading model: {str(e)}")

if __name__ == "__main__":
    test_load_model()
