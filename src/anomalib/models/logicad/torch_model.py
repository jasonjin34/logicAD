"""
This is official Logic Anomaly Detection with LLM implementation by using Pytorch, Pytorch-Lightning and
anomalib as general framework
"""
import torch.nn.functional as F
from torch import Tensor, nn

from .utils import init_json, update_json

from .text_prompt import (
    TEXT_EXTRACTOR_PROMPTS,
    TEXT_SUMMATION_PROMPTS,
)

from anomalib.utils.llm import (
    cos_sim,
    img2text,
    txt2sum,
    txt2embedding
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
        img2txt_db: str ="./dataset/loco.json",
        model_embedding: str = "text-embedding-3-large"
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.category = category
        self.model_vlm = model_vlm
        self.model_embedding = model_embedding
        self.reference_embedding = None
        self.reference_summation = None
        self.img2txt_db_path = img2txt_db
        self.img2txt_db_dict: dict = init_json(img2txt_db)
    
    def init_reference(self, reference_summation, reference_embedding):
        self.reference_summation = reference_summation
        self.reference_embedding = reference_embedding
    
    def text_extraction(self, image_path):
        """
        Extract text from the image
        """
        prompt = TEXT_EXTRACTOR_PROMPTS[self.category]
        if image_path in self.img2txt_db_dict:
            text = self.img2txt_db_dict[image_path]
        else:
            text = img2text(image_path, self.api_key, query=prompt, model=self.model_vlm)
            text = text["choices"][0]["message"]["content"]
            self.img2txt_db_dict[image_path] = text
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
    
    def build_memory_bank(self, reference_images):
        pass

    def forward(self, x: Tensor) -> Tensor:
        # x is image path
        if isinstance(x, list):
            x = x[0]
        template = self.reference_summation[0]
        text_summation = self.text_summation(x, template=template)
        print(text_summation)
        embedding = self.text_embedding(input_text=text_summation)
        # calculate the similarity
        score = 1 - cos_sim(embedding, self.reference_embedding[0])
        return score