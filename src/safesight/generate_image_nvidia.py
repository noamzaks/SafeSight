import requests
import os
import base64

invoke_url = "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-medium"

headers = {
    "Authorization": f"Bearer {os.environ['API_KEY']}",
    "Accept": "application/json",
}

payload = {
    "prompt": "realistic image of a road with cars",
    "cfg_scale": 5,
    "aspect_ratio": "16:9",
    "seed": 0,
    "steps": 50,
    "negative_prompt": ""
}

response = requests.post(invoke_url, headers=headers, json=payload)

response.raise_for_status()
response_body = response.json()
with open("generated_image.png", "wb") as f:
    f.write(base64.b64decode(response_body["image"]))
