from transformers import pipeline

# PLoad the GPT-2 model for text generation
summarizer = pipeline("text-generation", model="gpt2")

# Example text to summarize
your_text_here = ""

# Prepare the text for summarization
text_to_summarize = "TL;DR: " + your_text_here

# Generate the summary
summary = summarizer(text_to_summarize, max_length=100, num_return_sequences=1)[0]['generated_text']

# Print the summary
print(summary)