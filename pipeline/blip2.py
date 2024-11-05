from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering
from PIL import Image

image = Image.open("path_to_your_image.jpg")

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip2-opt-2.7b")

inputs = processor(images=image, text="What is happening in the image?", return_tensors="pt")

outputs = model.generate(**inputs)

generated_text = processor.decode(outputs[0], skip_special_tokens=True)
print("Generated Answer:", generated_text)