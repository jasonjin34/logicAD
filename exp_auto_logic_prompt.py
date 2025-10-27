from openai import OpenAI
import json
from formal_prompts_spec import PROMPT0, RULE, natural_normal_spec

# json_path = "datasets/breakfast_box/img2text.json"
# json_repo = "/Users/fengqihui/Documents/GitHub/logicAD/datasets/repo_img2text/"
# json_path = "datasets/juice_bottle/img2text_new.json"

API_KEY = ""  # TODO: Add your API key here or use environment variable

def txt2formal(
    api_key,
    model="gpt-4o",
    max_token=300,
    **prompt,
):
    """
    use openai to get the formalization of the text
    """
    # api_key = key_extraction(api_key)
    client = OpenAI(api_key=api_key)

    prompt0 = prompt['prompt']
    syn_rules = prompt['syn_rules']

    if prompt.get('k_shot', True):
        k_shot = prompt['k_shot']
    else:
        k_shot = ""
    
    if prompt.get('query', True):
        query = prompt['query']
    else:
        query = ""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are an AI assistant that helps people find information."}
                    ],
                },
            {
                "role": "user",
                "content": prompt0 + syn_rules + k_shot + query,
            }
        ],
        max_tokens=max_token,
        top_p=0.1
    )

    try:
        output = response.choices[0].message.content
    except Exception as e:
        output = None
    return output

# img_desc = "The left side of the image contains three fresh fruits: two tangerines or mandarins and one nectarine or peach. The right side of the image contains a mix of dried banana chips and almonds."
# one_shot = "An example is as follows:\nTEXT: The image shows a food tray with two compartments. \n\n- On the left side of the image, there are three pieces of fruit: one peach and two mandarins.\n- On the right side of the image, there is a mixture of granola, slices of dried banana, and some almonds.\nFORMULA: left(peach,1)\nleft(mandarin,2)\nright(granola,irrel)\nright(dried_banana,irrel)\nright(almond,irrel)\nNow output the formulae for another description:\nTEXT: "

# two_shot = "Some examples are as follows:\nTEXT: On the left side of the image, there is one apple and two mandarins (can also be oranges or tangerines, clementines). On the right side of the image, there is granola, banana chips (or dried banana chip), and almonds.\nFORMULA: \nleft(apple,1)\nleft(mandarin,2) OR left(orange,2) OR left(tangerine,2) OR left(clementine,2)\nright(granola,irrel)\nright(banana_chip,irrel) OR right(dried_banana_chip,irrel)\nright(amond,irrel)\n\nTEXT:The image shows a food tray with two compartments. \n\n- On the left side of the image, there are three pieces of fruit: one peach and two mandarins.\n- On the right side of the image, there is a mixture of granola, slices of dried banana, and some almonds.\nFORMULA: left(peach,1)\nleft(mandarin,2)\nright(granola,irrel)\nright(dried_banana,irrel)\nright(almond,irrel)\n\nNow output the formulae for another description:\nTEXT: "


# img_desc = "On the left side of the image, there are three tangerines and one apple. \n\nOn the right side of the image, there is a combination of granola, almonds, and dried banana chips."

# test_tag = 'splicing_connectors'
test_tag = 'breakfast_box'
# json_path = json_repo + '{}_img2text.json'.format(test_tag)
# with open(json_path, 'r') as jfile:
#     data = json.load(jfile)

# output = {}

natural_norm = natural_normal_spec[test_tag]
rlt = txt2formal(api_key=API_KEY, prompt=PROMPT0, syn_rules = RULE, query=natural_norm, k_shot="")
print(natural_norm + '\n' + rlt + '\n')

