import requests, os
import time

prompt = "You can only respond yes or no. Is there a car crash?"
directory = r"./zaksaset 2/zaksaset"
model_api = "https://ai.api.nvidia.com/v1/vlm/community/llava16-34b"

print(model_api)
print(prompt)

DEFAULT_HEADERS = {
  "Authorization": f"Bearer nvapi-{os.environ["API_KEY"]}",
  "Accept": "application/json"
}

for subfolder in ["accident", "nonaccident"]:
    print(subfolder)
    for filename in os.listdir(os.path.join(directory, subfolder)):
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
            print(filename, response["choices"][0]["message"]["content"])
        except:
            print(filename, response)

        response = requests.delete(f"https://api.nvcf.nvidia.com/v2/nvcf/assets/{asset_id}", headers=DEFAULT_HEADERS)

        time.sleep(2)
