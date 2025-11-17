import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Phi2ChatGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Phi-2 Offline Chat")
        self.root.geometry("700x500")

        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.create_widgets()
        threading.Thread(target=self.load_model, daemon=True).start()

    def create_widgets(self):
        self.chat_display = scrolledtext.ScrolledText(self.root, state='disabled', wrap=tk.WORD, font=("Consolas", 11))
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        input_frame = ttk.Frame(self.root)
        input_frame.pack(fill=tk.X, padx=10, pady=(0,10))

        self.prompt_entry = ttk.Entry(input_frame, font=("Consolas", 11))
        self.prompt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,10))
        self.prompt_entry.bind('<Return>', lambda event: self.on_send())

        self.send_btn = ttk.Button(input_frame, text="Send", command=self.on_send)
        self.send_btn.pack(side=tk.RIGHT)

        self.status_label = ttk.Label(self.root, text="Loading model... Please wait.")
        self.status_label.pack(pady=(0,10))

    def load_model(self):
        self.append_chat("[System] Loading Phi-2 model, please wait...")
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/phi-2",
            trust_remote_code=True
        )
        self.status_label.config(text="Model loaded. Enter your prompt below.")
        self.append_chat("[System] Model loaded. You can now chat!")

    def on_send(self):
        prompt = self.prompt_entry.get().strip()
        if not prompt or self.model is None:
            return
        self.append_chat(f"[You] {prompt}")
        self.prompt_entry.delete(0, tk.END)
        threading.Thread(target=self.generate_and_display, args=(prompt,), daemon=True).start()

    def generate_and_display(self, prompt):
        self.status_label.config(text="Generating response...")
        response = self.generate_response(prompt)
        self.append_chat(f"[Phi-2] {response}")
        self.status_label.config(text="Ready.")

    def generate_response(self, prompt, max_length=256):
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
        input_ids = inputs.input_ids.to(self.device)
        self.model.to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the response for clarity
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        return response

    def append_chat(self, message):
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, message + "\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state='disabled')

def main():
    root = tk.Tk()
    app = Phi2ChatGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
