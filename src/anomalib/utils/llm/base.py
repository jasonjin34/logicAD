# this is the utils function for the LLM based model and algorithm
import base64
import requests
import json
import os
import openai
from openai import OpenAI

from numpy import dot
import numpy as np
from numpy.linalg import norm

def cos_sim(a, b):
    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim

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
  

def txt2embedding(
	input_text= "",
	api_key=None,
	model="text-embedding-3-large",
):
	"""
	use openai to get the embedding of the text
	"""
	client = OpenAI(api_key=key_extraction(api_key))

	response = client.embeddings.create(
	    input=input_text,
	    model=model
	)
	return response.data[0].embedding

def txt2sum(
    input_text= "",
    few_shot_message="",
    api_key=None,
    model="gpt-4o",
    system_message="You are a helpful assistant designed to output JSON.",
    max_token=100,
):
    """
    use openai to get the summarization of the text
    """
    api_key = key_extraction(api_key)
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
      model=model,
      response_format={ "type": "json_object" },
      messages=[
        {
            "role": "system", 
            "content": system_message,
         },
        {"role": "user", "content": f"Can you summarize as following format, {few_shot_message}? {input_text}"},
      ]
    )
    output = response.choices[0].message.content
    print(output)
    return output
    
  
if __name__ == "__main__":
    api_key = "/home/erjin/git/Logic_AD_LLM/keys/gpt.json" 
    good_img = "/home/erjin/git/LocoAD/data/MVTec_LOCO/breakfast_box/test/good/007.png"
    test_img = "/home/erjin/git/LocoAD/data/MVTec_LOCO/breakfast_box/test/logical_anomalies/033.png" 
    test_img2 = "/home/erjin/git/Logic_AD_LLM/datasets/MVTec_LOCO/breakfast_box/test/logical_anomalies/038.png"
    test_img3 = "/home/erjin/git/Logic_AD_LLM/datasets/MVTec_LOCO/breakfast_box/test/logical_anomalies/064.png"
    test_good = "/home/erjin/git/Logic_AD_LLM/datasets/MVTec_LOCO/breakfast_box/test/good/011.png"
    query = "what is on the left side of image? and what is on the right side of image?"
    # test_text = img2text(test_img, api_key, query=query)["choices"][0]["message"]["content"]
    # test_text2 = img2text(test_img2, api_key, query=query)["choices"][0]["message"]["content"]
    test_bad = img2text(test_img, api_key, query=query)["choices"][0]["message"]["content"]
    test_good = img2text(test_good, api_key, query=query)["choices"][0]["message"]["content"]
    train= img2text(good_img, api_key, query=query)["choices"][0]["message"]["content"]


    train = txt2sum(
        input_text=train,
        few_shot_message="number objects, ...",
        api_key=api_key
    )

    test_bad = txt2sum(
        input_text=test_bad,
        few_shot_message="number objects, ...",
        api_key=api_key
    )

    test_good = txt2sum(
        input_text=test_good,
        few_shot_message="number objects, ...",
        api_key=api_key
    )
    

    train_embedding = txt2embedding(input_text=train, api_key=api_key)
    test_bad = txt2embedding(input_text=test_bad, api_key=api_key)
    test_good = txt2embedding(input_text=test_good, api_key=api_key)

    print(f"cosine similarity between train and bad image {cos_sim(train_embedding, test_bad)}")
    print(f"cosine similarity between train and good test image {cos_sim(train_embedding, test_good)}")


    # template = "The image contains {object} on the left side and {object} on the right side."

    # print("generting good text summary \n")
    # print(good_text)
    # template = txt2sum(
    #     input_text=good_text,
    #     few_shot_message=str(template),
    #     api_key=api_key
    # )

    # # print("generting testing text summary \n")
    # # print(test_text)
    # # txt2sum(
    # #     input_text=test_text,
    # #     few_shot_message=str(template),
    # #     api_key=api_key
    # # )

    # # print("generting testing text summary \n")
    # # print(test_text2)
    # # txt2sum(
    # #     input_text=test_text2,
    # #     few_shot_message=str(template),
    # #     api_key=api_key
    # # )

    # print("generting testing text summary \n")
    # print(test_text3)
    # txt2sum(
    #     input_text=test_text3,
    #     few_shot_message=str(template),
    #     api_key=api_key
    # )