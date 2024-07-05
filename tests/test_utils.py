"""Test for the utils functions"""

import pytest
import os
from anomalib.utils.llm.base import img2text

def test_img2text():
    image_path = "./tests/test_data/few_pushping.png"
    api_key = "./keys/gpt.json"
    output = img2text(image_path, api_key, model="gpt-4o", query="How many pushpins are there?")
    print(output)
    assert output is not None