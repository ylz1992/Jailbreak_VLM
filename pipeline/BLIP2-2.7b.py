import requests

API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip2-opt-2.7b"
headers = {"Authorization": "Bearer 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"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

output = query("./image/N_0/2_1.png")
print(output)