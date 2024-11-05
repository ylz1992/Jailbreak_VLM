import numpy as np
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel

# Initialize the CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name).to('cuda')

# Define constants
NUM_SAMPLES = 5
PROMPT_FILE = './data/Prompts_For_ConceptVector/Violence_30.csv'
OUTPUT_PATH = './data/concept_vector_violence.npy'

def load_prompts(file_path, violence_column, no_violence_column):
    df = pd.read_csv(file_path)
    violence_prompts = df[violence_column].tolist()
    no_violence_prompts = df[no_violence_column].tolist()
    return violence_prompts, no_violence_prompts

def encode_prompts(prompts):
    embeddings = []
    for prompt in prompts:
        repeated_prompt = [prompt] * NUM_SAMPLES
        inputs = processor(text=repeated_prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
        inputs = inputs.to('cuda')  # Move inputs to CUDA
        with torch.no_grad():
            embed = model.get_text_features(**inputs)
        embeddings.extend(embed.cpu().numpy())  # Move to CPU and store
    return np.array(embeddings)

def calculate_concept_vector(violence_embeddings, no_violence_embeddings):
    return np.mean(violence_embeddings - no_violence_embeddings, axis=0)

def save_concept_vector(concept_vector, output_path):
    np.save(output_path, concept_vector)
    print(f"Concept vector saved successfully at {output_path}")

def main():
    # Load prompts
    violence_prompts, no_violence_prompts = load_prompts(PROMPT_FILE, 'prompt', 'prompt1')
    
    # Encode prompts
    violence_embeddings = encode_prompts(violence_prompts)
    no_violence_embeddings = encode_prompts(no_violence_prompts)
    
    # Calculate concept vector
    concept_vector = calculate_concept_vector(violence_embeddings, no_violence_embeddings)
    
    # Save concept vector
    save_concept_vector(concept_vector, OUTPUT_PATH)

if __name__ == "__main__":
    main()