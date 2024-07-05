# helpfunction to build classifier
import torch
from torch import Tensor
from torch.nn import functional as F
from open_clip import SIMPLE_IMAGENET_TEMPLATES
from text_prompt import (
    STATE_LEVEL_ABNORMAL_PROMPTS,
    STATE_LEVEL_NORMAL_PROMPTS,
    TEMPLATE_LEVEL_PROMPTS
)


def text_encoder(
    text: Tensor,
    encoder_init,
    normalize: bool = True,
):
    model = encoder_init["clip_model"]
    text_features = model.encode_text(text)
    if normalize:
        text_features = F.normalize(text_features, dim=-1, p=2)
    return text_features


def build_text_classifier(
    encoder_init,
    category: str = None,
    background: bool = True
):
    tokenizer = encoder_init["tokenizer"]

    def _process_template(input_list):
        text = []
        if isinstance(input_list, list):
            for input in input_list:
                for template in SIMPLE_IMAGENET_TEMPLATES:
                    text.append(template(input))
        else:
            for template in SIMPLE_IMAGENET_TEMPLATES:
                text.append(template(input_list))
        texts = tokenizer(text)
        class_embeddings = text_encoder(texts, encoder_init)
        mean_class_embeddings = torch.mean(class_embeddings, dim=0, keepdim=True)
        mean_class_embeddings = F.normalize(mean_class_embeddings, dim=-1)
        return mean_class_embeddings

    background_template = [
        "write background",
        "noisy white background",
        "black background",
        "noisy white background",
        "white texture background"
    ]

    if category:
        text_features = _process_template(category)
        if background:
            background_features = _process_template(background_template)
            text_features = torch.cat(
                [text_features, background_features], dim=0)
        return text_features
    else:
        # no category, return embedding of empty string
        print("Category None, making embedding for empty string")
        texts = tokenizer([""])
        class_embeddings = text_encoder(texts, encoder_init)
        class_embeddings = F.normalize(class_embeddings, dim=-1)
        return class_embeddings
        

def build_ad_text_classifier(
    encoder_init, 
    category: str = "screw"
):
    tokenizer = encoder_init["tokenizer"]
    def _process_template(state_level_templates):
        text = []
        for template in TEMPLATE_LEVEL_PROMPTS:
            for state_template in state_level_templates:
                text.append(template(state_template(category)))

        texts = tokenizer(text)
        class_embeddings = text_encoder(texts, encoder_init)
        mean_class_embeddings = torch.mean(class_embeddings, dim=0, keepdim=True)
        mean_class_embeddings = F.normalize(mean_class_embeddings, dim=-1)
        return mean_class_embeddings

    normal_text_embedding = _process_template(STATE_LEVEL_NORMAL_PROMPTS)
    abnormal_text_embedding = _process_template(STATE_LEVEL_ABNORMAL_PROMPTS)
    text_features = torch.cat(
        [normal_text_embedding, abnormal_text_embedding], dim=0)
    return text_features
