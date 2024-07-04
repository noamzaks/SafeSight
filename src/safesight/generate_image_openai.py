# only for the rich - https://openai.com/chatgpt/pricing/

from openai import OpenAI
import os
import requests

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

image_prompt = "Realistic image of a road with cars"


response = client.images.generate(
  model="dall-e-2",
  prompt=image_prompt,
  size="1024x1024",
  quality="standard",
  n=1,
)

image = requests.get(response.data[0].url).content
with open("openai_generated_image.png", "wb") as f:
    f.write(image)
