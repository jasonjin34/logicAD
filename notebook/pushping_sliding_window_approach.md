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
path = "/home/erjin/git/Logic_AD_LLM/datasets/MVTec_Loco/mini/pushpins/test/logical_anomalies/030.png"
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
boxes, _, _ = get_bbx_from_query(image=trans_img, model=model)
```

```python
# convert boundingbox to 
```

```python
patches = patch_extraction_from_box(img, boxes=boxes)

# verify the extracted pushpin image
display_image(patches[5])
```

```python
# extract text features for each patch
img2text(patches[5], api_key=api_key, query="are different pushpins divided by plastic wall?, or only contain one pushpin?")
```

```python
## detection effective checking

good_path = "/home/erjin/git/Logic_AD_LLM/datasets/MVTec_Loco/original/pushpins/test/good"
bad_path = "/home/erjin/git/Logic_AD_LLM/datasets/MVTec_Loco/original/pushpins/test/logical_anomalies/"


import os
import glob

total_pathes = glob.glob(os.path.join(bad_path, "*"))
```

```python
for p in total_pathes[:2]:
    print(p)
    img, trans_img = dino_image_transform(path)
    boxes, logits, _ = get_bbx_from_query(image=trans_img, model=model)
    print(len(boxes), logits)
```

```python

```
