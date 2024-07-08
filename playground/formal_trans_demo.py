from openai import OpenAI

client = OpenAI(api_key="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")

prompt0 = "Given the description of an image, please output a formal specification as a set of (propositional) formulae. "
syn_rules = """Some syntactical rules are to be followed:\n
1. Each line consists of only one piece of fact.\n
2. Predicates are named in terms of properties such as location, color, size etc. Connect words with underline. Use lowercases only.\n
3. Objects and quantities are given as arguments of the predicates (use * if the object is uncountable or the number is irrelevant). Connect words with underline and always use singular form\n
4. Logical connectives such as $AND$, $OR$ $NOT$ might be used.
5. Description and output should be given in form 'Text: (Description of the image)\nFormulae: (a set of logical formulae)'"""

# one_shot = "An example is as follows:\nText: On the left side of the image, there is one apple and two mandarins (can also be oranges or tangerines, clementines). On the right side of the image, there is granola, banana chips (or dried banana chip), and almonds.\nFormulae: left(apple,1)\nleft(mandarin,2)$OR$left(orange,2)$OR$left(tangerine,2)$OR$left(clementine,2)\nright(granola,*)\nright(banana_chip,*)$OR$right(dried_banana_chip,*)\nright(amond,*)\n"
one_shot = "An example is as follows:\nText:The image shows a food tray with two compartments. \n\n- On the left side of the image, there are three pieces of fruit: one peach and two mandarins.\n- On the right side of the image, there is a mixture of granola, slices of dried banana, and some almonds.\nFormulae: left(peach,1)\nleft(mandarin,2)\nright(granola,*)\nright(dried_banana,*)\nright(almond,*)\nNow output the formulae for another description:\nText: "

two_shot = "Some examples are as follows:\nText: On the left side of the image, there is one apple and two mandarins (can also be oranges or tangerines, clementines). On the right side of the image, there is granola, banana chips (or dried banana chip), and almonds.\nFormulae: left(apple,1)\nleft(mandarin,2) $OR$ left(orange,2) $OR$ left(tangerine,2) $OR$ left(clementine,2)\nright(granola,*)\nright(banana_chip,*) $OR$ right(dried_banana_chip,*)\nright(amond,*)\n\nText:The image shows a food tray with two compartments. \n\n- On the left side of the image, there are three pieces of fruit: one peach and two mandarins.\n- On the right side of the image, there is a mixture of granola, slices of dried banana, and some almonds.\nFormulae: left(peach,1)\nleft(mandarin,2)\nright(granola,*)\nright(dried_banana,*)\nright(almond,*)\n\nNow output the formulae for another description:\nText: "
# img_desc = "On the left side of the image, there are three tangerines and one apple. \n\nOn the right side of the image, there is a combination of granola, almonds, and dried banana chips."
# img_desc = "On the left side of the image, there is one apple and two mandarins (can also be oranges or tangerines, clementines). On the right side of the image, there is granola, banana chips (or dried banana chip), and almonds."
img_desc = "The left side of the image contains three fresh fruits: two tangerines or mandarins and one nectarine or peach. The right side of the image contains a mix of dried banana chips and almonds."
# img_desc = "On the left side of the image, there is a mix of granola, dried banana slices, and almonds. On the right side of the image, there are three tangerines."
# img_desc = "On the left side of the image, there are three pieces of fruit: one nectarine and two mandarins. On the right side of the image, there is a pile of banana chips mixed with some almonds."
# img_desc = "On the left side of the image, there are two tangerines and a peach (or nectarine). On the right side of the image, there is a mixture of dried banana chips and almonds."
# img_desc = "The image shows a food tray with two compartments. \n\n- On the left side of the image, there are three pieces of fruit: one peach and two mandarins.\n- On the right side of the image, there is a mixture of granola, slices of dried banana, and some almonds."
response = client.chat.completions.create(
    model="gpt-4o",  
    messages=[
        {
            "role":"user",
            "content": prompt0 + syn_rules +two_shot + img_desc,
        }
    ],
    max_tokens=150  
)


generated_text = response.choices[0].message.content
print(generated_text)
