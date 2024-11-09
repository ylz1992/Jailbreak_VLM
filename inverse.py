import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import numpy as np
import random
import csv

# Set specific GPUs to be used
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6,7"

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

# Initialize Distributed Environment
def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    dist.destroy_process_group()

# Define fitness function
def fitness(population, targetEmbed, model):
    dummy_tokens = torch.cat(population, 0).to(targetEmbed.device)
    with torch.no_grad():
        dummy_embed = model.module.get_text_features(dummy_tokens)
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
def generate_jailbreak_prompt(row, concept_vector_path, num_generations, processor, model, population_size, device):
    prompt = row.prompt
    text_input = processor(text=[prompt], return_tensors="pt", padding="max_length", max_length=77, truncation=True).to(device)
    
    with torch.no_grad():
        targetEmbed = model.module.get_text_features(**text_input) + cof * torch.from_numpy(np.load(concept_vector_path)).to(device)
    targetEmbed = targetEmbed.detach().clone()

    # Initialize population for GA on CUDA
    population = [
        torch.cat((
            torch.tensor([[49406]], device=device),
            torch.randint(1, 49406, (1, length), device=device),
            torch.tensor([[49407]], device=device).repeat(1, 76 - length)
        ), 1) for _ in range(population_size)
    ]

    for step in range(num_generations):
        score = fitness(population, targetEmbed, model)
        idx = np.argsort(score)
        population = [population[index] for index in idx][:population_size // 2]

        if step != num_generations - 1:
            new_population = crossover(population, crossoverRate)
            population = mutation(new_population, mutateRate)

    jailbreak_prompt = processor.tokenizer.decode(population[0][0][1:length + 1])
    return prompt, jailbreak_prompt

# Main function to run the process
def main(rank, world_size, num_generations=3000, num_nudity_prompts=50, num_violence_prompts=0):
    setup_distributed(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Initialize CLIP model and processor on the given rank device
    model = CLIPModel.from_pretrained(model_name).to(device)
    model = DDP(model, device_ids=[rank])
    processor = CLIPProcessor.from_pretrained(model_name)

    # Load and distribute dataset based on rank
    df = pd.read_csv('./data/unsafe-prompts4703.csv')
    nudity_samples = df[df['nudity_percentage'] > 50].head(num_nudity_prompts)
    violence_samples = df[(df['categories'].str.contains('violence')) & 
                          (df['nudity_percentage'] < 50) & 
                          (df['inappropriate_percentage'] > 50) & 
                          (df['hard'] == 1)].head(num_violence_prompts)
    
    # Split samples for nudity and violence by rank
    nudity_split = np.array_split(nudity_samples, world_size)
    violence_split = np.array_split(violence_samples, world_size)
    
    # Process nudity prompts for the current rank
    for _, row in nudity_split[rank].iterrows():
        original_prompt, P_nudity = generate_jailbreak_prompt(row, path_Nudity_vector, num_generations, processor, model, population_size, device)
        with open(f'./InvPrompt/Nudity_prompts_rank{rank}.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([original_prompt, P_nudity])

    # Process violence prompts for the current rank
    for _, row in violence_split[rank].iterrows():
        original_prompt, P_violence = generate_jailbreak_prompt(row, path_Violence_vector, num_generations, processor, model, population_size, device)
        with open(f'./InvPrompt/Violence_prompts_rank{rank}.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([original_prompt, P_violence])

    cleanup_distributed()

if __name__ == "__main__":
    world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)