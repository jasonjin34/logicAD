from openai import OpenAI
import json

json_path = "datasets/breakfast_box/img2text.json"
PROMPT0 = (
    "Given the description of an image, please output a formal specification as a set of (propositional) formulae. "
)

RULE = """Some syntactical rules are to be followed:\n
1. Each line consists of only one piece of fact.\n
2. Predicates are named in terms of properties such as location, color, size etc. Connect words with underline. Use lowercases only.\n
3. Objects and quantities are given as arguments of the predicates (use irrel if the object is uncountable or the number is irrelevant). Connect words with underline and always use singular form\n
4. Logical connectives such as AND, OR, or NOT might be used.
5. Description is given in form 'TEXT: (Description of the image)\n' and output should be in form 'FORMULA: (a set of logical formulae)\n'."""

API_KEY = "sk-proj-mHskLBkbfFVMrlUQp4zoT3BlbkFJ9MicDBooQGIBm4JnVk3o"

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
                "role": "user",
                "content": prompt0 + syn_rules + k_shot + query,
            }
        ],
        max_tokens=max_token,
    )

    try:
        output = response.choices[0].message.content
    except Exception as e:
        output = None
    return output

img_desc = "The left side of the image contains three fresh fruits: two tangerines or mandarins and one nectarine or peach. The right side of the image contains a mix of dried banana chips and almonds."
one_shot = "An example is as follows:\TEXT: The image shows a food tray with two compartments. \n\n- On the left side of the image, there are three pieces of fruit: one peach and two mandarins.\n- On the right side of the image, there is a mixture of granola, slices of dried banana, and some almonds.\nFORMULA: left(peach,1)\nleft(mandarin,2)\nright(granola,irrel)\nright(dried_banana,irrel)\nright(almond,irrel)\nNow output the formulae for another description:\nTEXT: "

two_shot = "Some examples are as follows:\nTEXT: On the left side of the image, there is one apple and two mandarins (can also be oranges or tangerines, clementines). On the right side of the image, there is granola, banana chips (or dried banana chip), and almonds.\nFORMULA: left(apple,1)\nleft(mandarin,2) OR left(orange,2) OR left(tangerine,2) OR left(clementine,2)\nright(granola,irrel)\nright(banana_chip,irrel) OR right(dried_banana_chip,irrel)\nright(amond,irrel)\n\nTEXT:The image shows a food tray with two compartments. \n\n- On the left side of the image, there are three pieces of fruit: one peach and two mandarins.\n- On the right side of the image, there is a mixture of granola, slices of dried banana, and some almonds.\nFORMULA: left(peach,1)\nleft(mandarin,2)\nright(granola,irrel)\nright(dried_banana,irrel)\nright(almond,irrel)\n\nNow output the formulae for another description:\nTEXT: "
# img_desc = "On the left side of the image, there are three tangerines and one apple. \n\nOn the right side of the image, there is a combination of granola, almonds, and dried banana chips."

with open(json_path, 'r') as jfile:
    data = json.load(jfile)

output = {}

print("size: " + str(len(data)))
for img_key in data:
    print(img_key)
    img_desc = data[img_key]
    rlt = txt2formal(api_key=API_KEY, prompt=PROMPT0, syn_rules = RULE, query=img_desc, k_shot=two_shot)
    output[img_key] = rlt
    print(img_desc + '\n' + rlt + '\n')

with open('datasets/formal/breakfast_box_v1.1.json', 'w') as ofile:
    json.dump(output, ofile)
    print("done")

