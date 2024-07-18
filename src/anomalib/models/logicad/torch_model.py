"""
This is official Logic Anomaly Detection with LLM implementation by using Pytorch, Pytorch-Lightning and
anomalib as general framework
"""
from torch import Tensor, nn
import numpy as np
from .utils import init_json, update_json 

from .text_prompt import (
    TEXT_EXTRACTOR_PROMPTS,
    TEXT_SUMMATION_PROMPTS,
)

from anomalib.utils.llm import (
    cos_sim,
    img2text,
    txt2sum,
    txt2embedding,
    load_gdino_model
)
from .sliding_window import (
    dino_image_transform,
    get_bbx_from_query,
    calculate_centroidmap_disance,
)

class LogicadModel(nn.Module):
    """
    LOGICAD Model is powered by GPT API for processing the image
    """
    def __init__(
        self,
        api_key: str,
        category: str,
        model_vlm: str = "gpt-4o",
        top_p = None,
        temp = None,
        max_token = 300,
        img_size=128,
        img2txt_db: str ="./dataset/loco.json",
        model_embedding: str = "text-embedding-3-large",
        sliding_window: bool = False,
        gdino_cfg: str= "swint",
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.category = category
        self.sliding_window = sliding_window
        self.img_size = img_size
        self.max_token = max_token
        self.top_p = top_p
        self.temp = temp
        self.model_vlm = model_vlm
        self.model_embedding = model_embedding
        self.reference_embedding = None
        self.reference_summation = None
        self.reference_img_features = None
        self.img2txt_db_path = img2txt_db
        self.img2txt_db_dict: dict = init_json(img2txt_db)

        # only activate groudingdino if we applying sliding window
        self.gdino_model = load_gdino_model(cfg=gdino_cfg) if sliding_window else None
    
    def init_reference(self, reference_summation, reference_embedding, reference_img_features):
        self.reference_summation = reference_summation
        self.reference_embedding = reference_embedding
        self.reference_img_features = reference_img_features
    
    def generate_centroid_points(self, image_path):
        if self.gdino_model is None:
            raise ValueError("Sliding window is not activated, please activate it first")
        _, trans_img = dino_image_transform(image_path)
        boxes, _, _ = get_bbx_from_query(
            trans_img, self.gdino_model, box_threshold=0.35, text_threshold=0.35, query=self.category)
        return boxes
     
    def text_extraction(self, image_path):
        """
        Extract text from the image
        """
        def text_retrival(image_path):
            text = img2text(
                image_path, 
                self.api_key, 
                query=prompt, 
                model=self.model_vlm,
                max_tokens=self.max_token,
                img_size=self.img_size,
                temperature=self.temp,
                top_p=self.top_p
            )
            text = text["choices"][0]["message"]["content"]
            return text 
        prompt = TEXT_EXTRACTOR_PROMPTS[self.category]
        if image_path in self.img2txt_db_dict:
            text = self.img2txt_db_dict[image_path]
        else:
            text = text_retrival(image_path) 
            self.img2txt_db_dict[image_path] = text_retrival(image_path)
            update_json(self.img2txt_db_path, self.img2txt_db_dict)
        return text
    
    def text_summation(self, image_path, template="",):
        """
        Summarize the text
        """
        text = self.text_extraction(image_path)
        if template == "":
            template = TEXT_SUMMATION_PROMPTS[self.category]
        
        summary = txt2sum(
            input_text=text,
            few_shot_message=template,
            api_key=self.api_key,
            model=self.model_vlm
        )
        return summary

    def text_embedding(
        self, 
        image_path = None,
        input_text = None,
    ):
        """
        Get the embedding of the text
        """
        if input_text is None:
            text = self.text_extraction(image_path)
        else:
            text = input_text
        embedding = txt2embedding(input_text=text, api_key=self.api_key, model=self.model_embedding)
        return embedding
    
    def genenerate_img_features_score(self, x):
        x_centroid_points = self.generate_centroid_points(x)
        score = []
        for ref_img in self.reference_img_features:
            score.append(calculate_centroidmap_disance(ref_img, x_centroid_points))
        return np.array(score).mean()

    def build_memory_bank(self, reference_images):
        pass

    def forward(self, x: Tensor) -> Tensor:
        if self.sliding_window:
            score = self.genenerate_img_features_score(x)
        else:
            if isinstance(x, list):
                x = x[0]
            template = self.reference_summation[0]
            text_summation = self.text_summation(x, template=template)
            print(text_summation)
            embedding = self.text_embedding(input_text=text_summation)
            # calculate the similarity
            score = 1 - cos_sim(embedding, self.reference_embedding[0])
        return score