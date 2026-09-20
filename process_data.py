import json
import os
from src.preprocessing import clean_text

print("Processing raw corpus...")
input_file = "data/raw/raw_corpus.jsonl"
output_file = "data/processed/clean_corpus.jsonl"

os.makedirs("data/processed", exist_ok=True)

cleaned_count = 0
with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        data = json.loads(line)
        raw_text = data.get("text", "")
        cleaned = clean_text(raw_text)
        if cleaned:
            fout.write(json.dumps({"text": cleaned}) + "\n")
            cleaned_count += 1

print(f"Success! Processed and saved {cleaned_count} documents to {output_file}")