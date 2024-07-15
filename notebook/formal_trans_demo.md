---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
%load_ext autoreload
%autoreload 2
    
from anomalib.utils.llm import txt2formal
from anomalib.models.logicad.text_prompt import PROMPT0, RULE
api = "../keys/gpt.json"
```

## Please modify the text_prompt in anomalib.models.logicad.text_prompt file

### https://github.com/jasonjin34/logicAD/blob/master/src/anomalib/models/logicad/text_prompt.py

```python
img_desc = "The left side of the image contains three fresh fruits: two tangerines or mandarins and one nectarine or peach. The right side of the image contains a mix of dried banana chips and almonds."
one_shot = "An example is as follows:\nText:The image shows a food tray with two compartments. \n\n- On the left side of the image, there are three pieces of fruit: one peach and two mandarins.\n- On the right side of the image, there is a mixture of granola, slices of dried banana, and some almonds.\nFormulae: left(peach,1)\nleft(mandarin,2)\nright(granola,*)\nright(dried_banana,*)\nright(almond,*)\nNow output the formulae for another description:\nText: "

two_shot = "Some examples are as follows:\nText: On the left side of the image, there is one apple and two mandarins (can also be oranges or tangerines, clementines). On the right side of the image, there is granola, banana chips (or dried banana chip), and almonds.\nFormulae: left(apple,1)\nleft(mandarin,2) $OR$ left(orange,2) $OR$ left(tangerine,2) $OR$ left(clementine,2)\nright(granola,*)\nright(banana_chip,*) $OR$ right(dried_banana_chip,*)\nright(amond,*)\n\nText:The image shows a food tray with two compartments. \n\n- On the left side of the image, there are three pieces of fruit: one peach and two mandarins.\n- On the right side of the image, there is a mixture of granola, slices of dried banana, and some almonds.\nFormulae: left(peach,1)\nleft(mandarin,2)\nright(granola,*)\nright(dried_banana,*)\nright(almond,*)\n\nNow output the formulae for another description:\nText: "
# img_desc = "On the left side of the image, there are three tangerines and one apple. \n\nOn the right side of the image, there is a combination of granola, almonds, and dried banana chips."
txt2formal(api_key=api, prompt=PROMPT0, syn_rules = RULE, query=img_desc, k_shot=one_shot)
```

```python

```
