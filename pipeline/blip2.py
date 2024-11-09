import torch
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering
from PIL import Image

# Load and prepare the image
image = Image.open("./image/2_2.png").convert('RGB')

# Load the processor and model, and move the model to a specific GPU (e.g., cuda:0)
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip2-opt-2.7b").to("cuda:0")

# Process inputs and explicitly move them to the same GPU
inputs = processor(images=image, text="how many dogs in this picture?", return_tensors="pt").to("cuda:0")

# Generate outputs with modified generation settings
with torch.no_grad():  # Disable gradients for inference
    outputs = model.generate(**inputs)

# Decode the outputs and print the generated text
generated_text = processor.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
print("Generated Answer:", generated_text)