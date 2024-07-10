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
# TODO finish drafted whole sliding windows approach to handling pushping number issues

from anomalib.utils.llm import img2text, txt2sum
from sliding_window import load_image, display_image, image_unfold, display_multiple_image

api_key = "/home/erjin/git/Logic_AD_LLM/keys/gpt.json"
path = "/home/erjin/git/Logic_AD_LLM/datasets/MVTec_Loco/mini/pushpins/test/logical_anomalies/023.png"
```

```python
display_image(load_image(path))
```

```python
img = load_image(path)
```

```python
display_image((image_unfold(img, 200,  pos=6)))
```

```python
# Test the performance on image patches

image_patches = image_unfold(img, 200, pos=-1)
```

```python
test_patch = image_patches[6]
```

```python
query = "describe the full-size compartments"
img2text(test_patch, api_key=api_key, query=query)
```

```python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from groundingdino.config import GroundingDINO_SwinB_cfg as gdino_cfg
from groundingdino.util.inference import (
    load_image,
    predict,
    annotate,
)
from anomalib.utils.llm import load_gdino_model
```

```python
# ckpt path 
ckpt_path = "/DATA1/llm-research/GroundingDino/groundingdino_swint_ogc.pth"
model = load_gdino_model(ckpt_path)
```

```python
img, transformed_img = load_image(path)
```

```python
TEXT_PROMPT = "pushpin"
BOX_TRESHOLD = 0.3
TEXT_TRESHOLD = 0.3

boxes, logits, phrases = predict(
    model=model, 
    image=transformed_img, 
    caption=TEXT_PROMPT, 
    box_threshold=BOX_TRESHOLD, 
    text_threshold=TEXT_TRESHOLD,
    device="cuda"
)
annotated_frame = annotate(image_source=img, boxes=boxes, logits=logits, phrases=phrases)
```

```python
Image.fromarray(annotated_frame)
```

```python
# show the divided conqure method, find all the pushpins location and then determine if the pusppings are near enough
```

```python
# convert box to patch
import torch
import numpy as np
def patch_extraction_from_box(image, box, patch=1):
    # assume the image would be tensor, need to change to numpy if necessary
    if isinstance(image, np.ndarray):
        w, h, _ = image.shape
        image = torch.tensor(image).permute(2, 0, 1)
    if w > h:
        patch = int(h / 3 * patch)
    else:
        patch = int(w / 3 * patch)
    # convert box to image dim
    # box format, center w, center h
    cx, cy = int(box[0] * h), int(box[1] * w)
    print(cx, cy)
    sx = cx - patch // 2
    ex = cx + patch // 2
    sy = cy - patch // 2
    ey = cy + patch // 2

    if sx < 0:
        sx = 0
    if sy < 0:
        sy = 0

    print(sx, ex, sy, ey)
    
    output_image = image[:,sy:ey, sx:ex]
    display_image(output_image)
    print(output_image.shape)
    return output_image 
```

```python
patch = patch_extraction_from_box(img, boxes[9])
```

```python
query = "are different pushpins divided by plastic wall?, or only contain one pushpin?"

for _ in range(5):
    text = img2text(patch, api_key=api_key, query=query, temperature=0.7, top_p=0.6)["choices"][0]["message"]["content"]
    print(text)
```

```python

```

```python

```

```python

```

```python

```

```python

```
