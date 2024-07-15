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
```

```python
# Design the overall pipeline for data proprocessing to the patch text extraction

category = "pushpin"
api_key = "/home/erjin/git/Logic_AD_LLM/keys/gpt.json"
path = "/home/erjin/git/Logic_AD_LLM/datasets/MVTec_Loco/original/pushpins/test/good/001.png"
```

```python
# get image transform and feed into GroundingDINO

from anomalib.models.logicad.sliding_window import (
    dino_image_transform, 
    patch_extraction_from_box, 
    get_bbx_from_query
)

from anomalib.utils.llm import load_gdino_model
from anomalib.utils.llm import img2text, txt2sum
from sliding_window import load_image, display_image, image_unfold, display_multiple_image
```

```python
# get the r bbx for the object
img, trans_img = dino_image_transform(path)
model = load_gdino_model(ckpt_path="/home/erjin/git/Logic_AD_LLM/datasets/GroundingDino/groundingdino_swint_ogc.pth")
boxes, logits, phrases = get_bbx_from_query(image=trans_img, model=model, box_threshold=0.35, text_threshold=0.35, query="pushpin")
```

```python
# convert boundingbox
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from groundingdino.config import GroundingDINO_SwinB_cfg as gdino_cfg
from groundingdino.util.inference import (
    load_image,
    predict,
    annotate,
)
annotated_frame = annotate(image_source=img, boxes=boxes, logits=logits, phrases=phrases)

print(len(boxes))

Image.fromarray(annotated_frame)

```

```python
## detection effective checking

good_path = "/home/erjin/git/Logic_AD_LLM/datasets/MVTec_Loco/mini//pushpins/test/good/"
bad_path = "/home/erjin/git/Logic_AD_LLM/datasets/MVTec_Loco/mini/pushpins/test/logical_anomalies/"


import os
import glob

total_pathes = glob.glob(os.path.join(bad_path, "*"))
```

```python
for p in total_pathes:
    print(p)
    _, trans_img = dino_image_transform(p)
    boxes, logits, _ = get_bbx_from_query(image=trans_img, model=model)
    print(len(boxes))
```

```python
# generating batch of text features 
path = "/home/erjin/git/Logic_AD_LLM/datasets/MVTec_Loco/original/pushpins/test/logical_anomalies/004.png"
img, trans_img = dino_image_transform(path)

boxes, logits, phrases = get_bbx_from_query(image=trans_img, model=model, box_threshold=0.35, text_threshold=0.35, query="pushpin")

patches = patch_extraction_from_box(img, boxes=boxes, patch=1.5)

text_features = []

def patch_text_extraction(patch, api_key=api_key, query="describe center compartment, as {object: number, wall between pushping: yes or no}"):
    text = img2text(patch, api_key=api_key, query=query, top_p=0.02, temperature=0.02)
    return text["choices"][0]["message"]["content"]

# extract text features for each patch
for idx, p in enumerate(patches):
    text = patch_text_extraction(p)
    print(text)
    text_features.append(text)
```

txt2sum(str(text_features), api_key=api_key,few_shot_message="only summarize the unique sentences")

```python
# verify the extracted pushpin image
display_image(patches[-1])
```

```python
idx = -2

query = "describe centre square compartment"#, as {push: number, more than one pushpin in same compartment: yes or no}"

def patch_text_extraction(patch, api_key=api_key, query=query):
    text = img2text(patch, api_key=api_key, query=query, top_p=0.05, temperature=0.05)
    return text["choices"][0]["message"]["content"]


print(patch_text_extraction(patches[idx]))
display_image(patches[idx])
```

```python
Image.fromarray(img)
```

```python

```
