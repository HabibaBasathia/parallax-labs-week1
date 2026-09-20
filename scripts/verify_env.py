import torch
import pandas as pd
import sentence_transformers
import chromadb
import spacy

print("--- ENVIRONMENT CHECK ---")
print("CUDA Available:", torch.cuda.is_available())
print("Pandas Version:", pd.__version__)
print("Sentence-Transformers loaded successfully!")
print("ChromaDB loaded successfully!")
print("Spacy loaded successfully!")