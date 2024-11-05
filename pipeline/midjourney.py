import pandas as pd
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
pipe.load_lora_weights("Jovie/Midjourney")

df = pd.read_csv('./InvPrompt/a.csv', header=None, names=["prompt1", "prompt2"])

output_dir = './image/'

for index, row in df.iterrows():
    prompt_n = row["prompt1"]  # normal sentence
    prompt_a = row["prompt2"]  # attack sentence
    
    image_n = pipe(prompt_n).images[0]
    image_a = pipe(prompt_a).images[0]
    
    index += 1  # Offset index to start from 1
    filename_n = f"{output_dir}{index}_n.png"
    filename_a = f"{output_dir}{index}_a.png"
    
    # Save the images
    image_n.save(filename_n)
    image_a.save(filename_a)
    
    print(f"Images saved as {filename_n} and {filename_a}")