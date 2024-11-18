import requests
import io
from PIL import Image
import os
import csv

API_URL = "https://api-inference.huggingface.co/models/Jovie/Midjourney"
headers = {"Authorization": "Bearer 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"}

# Directory to save the generated images
output_dir = './image/N_4'
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    # Check if the response is JSON (indicating an error or message)
    if response.headers.get("content-type") == "application/json":
        print("Error:", response.json())  # Print the error message
        return None
    return response.content

# Read each line in a.csv and process both sentences
with open('./InvPrompt/Nudity_prompts_rank4.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for line_number, row in enumerate(reader, start=1):
        if len(row) >= 2:
            sentence1, sentence2 = row[0].strip(), row[1].strip()  # Strip extra whitespace
            
            # Process each sentence in the row
            for i, sentence in enumerate([sentence1, sentence2], start=1):
                # Query the API with the sentence
                image_bytes = query({"inputs": sentence})
                
                # Save the image if generated successfully
                if image_bytes:
                    image = Image.open(io.BytesIO(image_bytes))
                    image_path = os.path.join(output_dir, f"{line_number}_{i}.png")
                    image.save(image_path)
                    print(f"Image saved to {image_path}")
                else:
                    print(f"Failed to generate an image for line {line_number}, sentence {i}")
        else:
            print(f"Line {line_number} does not have two sentences.")