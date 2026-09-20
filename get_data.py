from datasets import load_dataset
import json
import os

print("Downloading dataset from Hugging Face...")
# Using TinyStories as a clean, modern text dataset
dataset = load_dataset("roneneldan/TinyStories", split="train[:5000]")

# Ensure the raw data folder exists
os.makedirs("data/raw", exist_ok=True)

output_path = "data/raw/raw_corpus.jsonl"
print(f"Saving raw data to {output_path}...")

with open(output_path, "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps({"text": item["text"]}) + "\n")

print("Dataset downloaded and saved successfully!")