import numpy as np
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel

# Initialize the CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name).to('cuda')

# Constants
NUM_SAMPLES = 5
NUDITY_PROMPT_FILE = './data/Prompts_For_ConceptVector/Nudity_prompt.csv'
NO_NUDITY_PROMPT_FILE = './data/Prompts_For_ConceptVector/NoNudity_prompt.csv'
OUTPUT_PATH = './data/concept_vector_nudity.npy'
EXCLUDE_INDICES = [134, 15, 16, 147, 151, 35, 38, 174, 55, 62, 202, 204, 91, 93, 95, 96, 98, 106, 109]

# Load nudity prompts
df_nudity = pd.read_csv(NUDITY_PROMPT_FILE)
nudity_text = []
for _, row in df_nudity.iterrows():
    if row['clothes'] != 'None':
        if row['people'] == 'None':
            prompt = [f"{row['nudity']}, {row['location']}"] * NUM_SAMPLES
        else:
            prompt = [f"{row['nudity']} {row['people']}, {row['location']}"] * NUM_SAMPLES
    else:
        prompt = [f"{row['nudity']} {row['people']}, {row['location']}"] * NUM_SAMPLES

    # Tokenize and encode
    inputs = processor(text=prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True).to('cuda')
    with torch.no_grad():
        embed = model.get_text_features(**inputs)
    nudity_text.extend(embed.cpu().numpy())
nudity_text = np.array(nudity_text)

# Load no-nudity prompts
df_no_nudity = pd.read_csv(NO_NUDITY_PROMPT_FILE)
no_nudity_text = []
for _, row in df_no_nudity.iterrows():
    prompt = [f"{row['prompt']}"] * NUM_SAMPLES
    # Tokenize and encode
    inputs = processor(text=prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True).to('cuda')
    with torch.no_grad():
        embed = model.get_text_features(**inputs)
    no_nudity_text.extend(embed.cpu().numpy())
no_nudity_text = np.array(no_nudity_text)

# Remove failed indices
nudity_text = np.delete(nudity_text, EXCLUDE_INDICES, axis=0)
no_nudity_text = np.delete(no_nudity_text, EXCLUDE_INDICES, axis=0)

# Calculate and save concept vector
concept_vector = np.mean(nudity_text - no_nudity_text, axis=0)
np.save(OUTPUT_PATH, concept_vector)
print(f"Concept vector saved successfully at {OUTPUT_PATH}")