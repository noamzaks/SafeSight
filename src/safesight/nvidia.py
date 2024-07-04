import requests
import os
import time
import tqdm
import base64

import click

from safesight.cli import cli


@cli.group()
def nvidia():
    """
    Commands for interacting with the NVIDIA API.
    """


@nvidia.command()
@click.option(
    "--directory",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default="./zaksaset/zaksaset",
    show_default=True,
)
@click.option(
    "--prompt",
    type=str,
    default="Realistic image of a road with cars",
    show_default=True,
)
@click.option(
    "--model",
    type=str,
    default="community/llava16-34b",
    show_default=True,
)
def run_on_dataset(directory, prompt, model):
    assert "NVIDIA_API_KEY" in os.environ, "Please set the NVIDIA_API_KEY environment variable."

    model_api = f"https://ai.api.nvidia.com/v1/vlm/{model}"

    print(model_api)
    print(prompt.replace("\n", " "))

    DEFAULT_HEADERS = {
    "Authorization": f"Bearer {os.environ["NVIDIA_API_KEY"]}",
    "Accept": "application/json"
    }

    for subfolder in ["accident", "nonaccident"]:
        print(subfolder)
        for filename in tqdm.tqdm(sorted(os.listdir(os.path.join(directory, subfolder)))):
            if not filename.endswith(".png"):
                continue
            response = requests.post("https://api.nvcf.nvidia.com/v2/nvcf/assets", headers=DEFAULT_HEADERS | { "Content-Type": "application/json"}, json={
                "contentType": "image/png",
                "description": "potato"
            })
            response = response.json()
            asset_id = response["assetId"]
            upload_url = response["uploadUrl"]


            with open(os.path.join(directory, subfolder, filename), "rb") as f:
                image_contents = f.read()

            response = requests.put(upload_url, headers={ "Content-Type": "image/png", "x-amz-meta-nvcf-asset-description": "potato" }, data=image_contents)

            response = requests.post(model_api, headers=DEFAULT_HEADERS | {"NVCF-INPUT-ASSET-REFERENCES": asset_id}, json={"messages": [{"role": "user", "content": f'{prompt}. <img src="data:image/png;asset_id,{asset_id}" />' }]})
            response = response.json()
            try:
                print(filename, response["choices"][0]["message"]["content"].strip().replace("\n", " "))
            except:
                print(filename, response)

            response = requests.delete(f"https://api.nvcf.nvidia.com/v2/nvcf/assets/{asset_id}", headers=DEFAULT_HEADERS)

            time.sleep(2)


@nvidia.command()
@click.option(
    "--prompt",
    type=str,
    default="Realistic image of a road with cars",
    show_default=True,
)
@click.option(
    "--model",
    type=str,
    default="stabilityai/stable-diffusion-3-medium",
    show_default=True,
)
@click.option(
    "--output-path",
    type=click.Path(exists=False, dir_okay=False, file_okay=True),
    default="generated_image.png",
    show_default=True,
)
def generate_image(prompt, model, output_path):
    assert "NVIDIA_API_KEY" in os.environ, "Please set the NVIDIA_API_KEY environment variable."

    invoke_url = f"https://ai.api.nvidia.com/v1/genai/{model}"

    headers = {
        "Authorization": f"Bearer {os.environ["NVIDIA_API_KEY"]}",
        "Accept": "application/json",
    }

    payload = {
        "prompt": prompt,
        "cfg_scale": 5,
        "aspect_ratio": "16:9",
        "seed": 0,
        "steps": 50,
        "negative_prompt": ""
    }

    response = requests.post(invoke_url, headers=headers, json=payload)

    response.raise_for_status()
    response_body = response.json()
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(response_body["image"]))
