from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Choose your model (Uncomment the one you prefer)
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Llama 2
# MODEL_NAME = "tiiuae/falcon-7b"  # Falcon

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

# Create pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def jarvis_chat(prompt):
    result = generator(prompt, max_length=200, do_sample=True, temperature=0.7)
    return result[0]['generated_text']

if __name__ == "__main__":
    user_input = input("You: ")
    response = jarvis_chat(user_input)
    print("Jarvis:", response)
