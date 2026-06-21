from transformers import pipeline

print("Loading AI Model...")

# Load GPT text generation model
generator = pipeline("text-generation", model="gpt2")

print("Model Loaded Successfully!")

# User prompt
prompt = input("Enter a topic: ")

# Generate text
result = generator(
    prompt,
    max_length=100,
    num_return_sequences=1
)

print("\nGenerated Text:\n")

print(result[0]['generated_text'])
