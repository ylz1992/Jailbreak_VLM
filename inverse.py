from transformers import CLIPProcessor, CLIPModel
import torch
import pandas as pd
import numpy as np
import random
import csv

# Constants and configurations for GA
population_size = 200
generation = 3000
mutateRate = 0.25
crossoverRate = 0.5
length = 16  # For K = 77, set length = 75
cof = 3
path_Nudity_vector = './data/concept_vector_nudity.npy'
path_Violence_vector = './data/concept_vector_violence.npy'
model_name = "openai/clip-vit-base-patch32"
torch_device = 'cuda'  # Ensure CUDA usage

# Initialize the processor and model on CUDA
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name).to(torch_device)

# Load the dataset
df = pd.read_csv('./data/unsafe-prompts4703.csv')

# Define fitness function
def fitness(population, targetEmbed):
    dummy_tokens = torch.cat(population, 0).to(torch_device)
    with torch.no_grad():
        dummy_embed = model.get_text_features(dummy_tokens)
    losses = ((targetEmbed - dummy_embed) ** 2).sum(dim=1)
    return losses.cpu().numpy()

# Define crossover function
def crossover(parents, crossoverRate):
    new_population = []
    for i in range(len(parents)):
        new_population.append(parents[i])
        if random.random() < crossoverRate:
            idx = np.random.randint(0, len(parents))
            crossover_point = np.random.randint(1, length + 1)
            new_population.append(torch.cat((parents[i][:, :crossover_point], parents[idx][:, crossover_point:]), 1))
            new_population.append(torch.cat((parents[idx][:, :crossover_point], parents[i][:, crossover_point:]), 1))
    return new_population

# Define mutation function
def mutation(population, mutateRate):
    for i in range(len(population)):
        if random.random() < mutateRate:
            idx = np.random.randint(1, length + 1)
            value = np.random.randint(1, 49406)
            population[i][:, idx] = value
    return population

# Function to generate jailbreak prompt and return both P and original prompt
def generate_jailbreak_prompt(row, concept_vector_path, num_generations):
    prompt = row.prompt
    text_input = processor(text=[prompt], return_tensors="pt", padding="max_length", max_length=77, truncation=True).to(torch_device)
    
    with torch.no_grad():
        targetEmbed = model.get_text_features(**text_input) + cof * torch.from_numpy(np.load(concept_vector_path)).to(torch_device)
    targetEmbed = targetEmbed.detach().clone()

    # Initialize population for GA on CUDA
    population = [
        torch.cat((
            torch.tensor([[49406]], device=torch_device),  # CLIP's start token
            torch.randint(1, 49406, (1, length), device=torch_device),  # Random tokens for GA
            torch.tensor([[49407]], device=torch_device).repeat(1, 76 - length)  # CLIP's end token padding
        ), 1) for _ in range(population_size)
    ]

    # Run GA to refine the prompt with the specified number of generations
    for step in range(num_generations):
        score = fitness(population, targetEmbed)
        idx = np.argsort(score)
        population = [population[index] for index in idx][:population_size // 2]

        if step != num_generations - 1:
            new_population = crossover(population, crossoverRate)
            population = mutation(new_population, mutateRate)

    # Decode and return original and jailbreak prompt
    jailbreak_prompt = processor.tokenizer.decode(population[0][0][1:length + 1])
    return prompt, jailbreak_prompt

# Main function to run the process with a limited number of prompts for testing
def main(num_generations=3000, num_nudity_prompts=10, num_violence_prompts=10):
    # Filter for nudity and violence prompts
    nudity_samples = df[df['nudity_percentage'] > 50].head(num_nudity_prompts)
    violence_samples = df[(df['categories'].str.contains('violence')) & 
                          (df['nudity_percentage'] < 50) & 
                          (df['inappropriate_percentage'] > 50) & 
                          (df['hard'] == 1)].head(num_violence_prompts)
    
    # Process nudity prompts
    for _, row in nudity_samples.iterrows():
        original_prompt, P_nudity = generate_jailbreak_prompt(row, path_Nudity_vector, num_generations)
        with open('./InvPrompt/Nudity_prompts.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([original_prompt, P_nudity])

    # Process violence prompts
    for _, row in violence_samples.iterrows():
        original_prompt, P_violence = generate_jailbreak_prompt(row, path_Violence_vector, num_generations)
        with open('./InvPrompt/Violence_prompts.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([original_prompt, P_violence])

# Entry point
if __name__ == "__main__":
    # You can set the number of prompts to process here
    main(num_generations=3000, num_nudity_prompts=10, num_violence_prompts=10)