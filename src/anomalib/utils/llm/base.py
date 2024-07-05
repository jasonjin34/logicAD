# this is the utils function for the LLM based model and algorithm
import base64
import requests
import json
import os

def key_extraction(file_path):
    if os.path.exists(file_path) is False:
        raise FileNotFoundError(f"GPT api key file not found: {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)
        key = data["gpt"]
    return key

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def img2text(
    image_path,
    api_key,
    model="gpt-4o",
    query="How many pushpins are there?",
):
    """
    image text extracton using LLM model, so far only support gpt-4o model
    # TODO 
    # add other LLM model, particular for some small model for the ablation study
    """

    api_key = key_extraction(api_key)
    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
    }

    payload = {
      "model": model,
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"{query}"
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      "max_tokens": 300
    }
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        output = response.json()
    except Exception as e:
        print(f"Have problem in extracting text, Error in the request: {e}")
        output = None
    return output

