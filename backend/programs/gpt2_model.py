from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the GPT-2 model and tokenizer
def load_model():
    model_name = "gpt2" # GPT-2 model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model

# Generate recommendations based on the form data
def generate_recommendations(form_data, probability):
    tokenizer, model = load_model()  # Fixed the order here

    # Define the prompt to evaluate
    prompt = f"""
    Here is a random fact about cancer:
    
    Did you know that...?
    """

    # Encode the prompt using the tokenizer
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generate a response
    outputs = model.generate(inputs, max_length=1000, num_return_sequences=1, no_repeat_ngram_size=2)

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text