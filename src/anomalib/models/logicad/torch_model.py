"""
This is official Logic Anomaly Detection with LLM implementation by using Pytorch, Pytorch-Lightning and
anomalib as general framework
"""
from torch import nn
import numpy as np
import os
from .utils import init_json, update_json 

from .text_prompt import (
    TEXT_EXTRACTOR_PROMPTS,
    TEXT_SUMMATION_PROMPTS,
    TEXT_EXTRACTOR_PROMPTS_SA,
    TEXT_SUMMATION_PROMPTS_SA,
)

from anomalib.utils.llm import (
    cos_sim,
    img2text,
    txt2sum,
    txt2txt,
    txt2embedding,
    load_gdino_model,
    load_model,
)

from .sliding_window import (
    dino_image_transform,
    get_bbx_from_query,
    calculate_centroidmap_disance,
    patch_extraction_from_box
)

class LogicadModel(nn.Module):
    """
    LOGICAD Model is powered by GPT API for processing the image
    """
    def __init__(
        self,
        api_key: str,
        category: str,
        model_vlm: str = "gpt-4o-az",
        model_llm: str = "gpt-4o-az",
        top_p = None,
        temp = None,
        max_token = 300,
        img_size=128,
        img2txt_db: str ="./dataset/loco.json",
        model_embedding: str = "text-embedding-3-large-az",
        seed: int = 42,
        sliding_window: bool = False,
        croping_patch: bool = False,
        gdino_cfg: str= "swint",
        device: str = "cuda:1",
        wo_summation: bool = False,
        threshold: float = 0.14,
        num_text_extraction: int = 1,
        sa: bool = False,
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.category = category
        self.sliding_window = sliding_window
        self.img_size = img_size
        self.max_token = max_token
        self.top_p = top_p
        self.temp = temp
        self.cropping_patch = croping_patch
        self.num_text_extraction = num_text_extraction
        self.threshold = threshold
        self.seed = seed
        self.wo_summation= wo_summation
        self.model_vlm = model_vlm
        self.model_llm = model_llm
        self.model_embedding = model_embedding
        self.reference_embedding = None
        self.reference_summation = None
        self.reference_img_features = None
        self.reference_img_paths = None
        self.abnormal_ref_embedding = None
        self.text_extractor_prompt = TEXT_EXTRACTOR_PROMPTS_SA if sa else TEXT_EXTRACTOR_PROMPTS
        self.text_summation_prompt = TEXT_SUMMATION_PROMPTS_SA if sa else TEXT_SUMMATION_PROMPTS

        if model_vlm not in ["gpt-4o", "gpt-4o-turbo", "gpt-4o-az"]:
            self.model_vlm_pipeline = load_model(model_vlm, "image-to-text", device=device)
            basename = model_vlm + "_" + os.path.basename(img2txt_db)
            self.img2txt_db_path = os.path.join(os.path.dirname(img2txt_db), basename)
            self.img2txt_db_dict: dict = init_json(self.img2txt_db_path)
        else:
            self.img2txt_db_path = img2txt_db
            self.img2txt_db_dict: dict = init_json(img2txt_db)
            self.model_vlm_pipeline = None

        self.gdino_model = load_gdino_model(cfg=gdino_cfg)
    
    def init_reference(
        self, 
        reference_summation, 
        reference_embedding, 
        reference_img_features, 
        reference_img_paths,
    ):
        self.reference_summation = reference_summation
        self.reference_embedding = reference_embedding
        self.reference_img_features = reference_img_features
        self.reference_img_paths = reference_img_paths

        if self.wo_summation:
            self.reference_embedding = self.text_embedding(input_text=self.reference_img_paths[0])
    
    def generate_centroid_points(self, image_path):
        if self.gdino_model is None:
            raise ValueError("Sliding window is not activated, please activate it first")
        
        _, trans_img = dino_image_transform(image_path)
        boxes, _, _ = get_bbx_from_query(
            trans_img, 
            self.gdino_model, 
            box_threshold=0.35, 
            text_threshold=0.35, 
            query=self.category
        )
        return boxes[:2]
     
    def text_extraction(self, image_path):
        """
        Extract text from the image
        """
        def text_retrival(image_path, prompt):
            if self.cropping_patch:
                text = ""
                patches, img = self.generate_crop_img(image_path)
                print("number of patches", len(patches))
                for p in patches:
                    patch_text_list = []
                    for _ in range(5):
                        patch_text = img2text(
                            p, 
                            self.api_key, 
                            query=prompt, 
                            model_name=self.model_vlm,
                            model=self.model_vlm_pipeline,
                            max_tokens=self.max_token,
                            img_size=self.img_size,
                            temperature=self.temp,
                            top_p=self.top_p,
                            seed=self.seed
                        )
                        patch_text_list.append(patch_text)
                    test_list_str = str(patch_text_list)
                    summa_query = "select the most frequent text from the list, only give the text"
                    text_path = txt2txt(test_list_str, query=summa_query, api_key=self.api_key, model=self.model_llm).replace('"', '')
                    text = text +  " patch description: " + text_path + "."
                if self.category != "screw_bag":
                    summa_query = "select the unique text from the list, only give the unique text"
                    text = txt2txt(text, query=summa_query, api_key=self.api_key, model=self.model_llm).replace('"', '') 
                print(text)
            else:
                # if self.category == "screw_bag":
                #     patches, _ = self.generate_crop_img(image_path)
                #     num_guide = f"Given total number of washer and nut is {len(patches)}l "
                #     prompt = num_guide + prompt
                text = img2text(
                    image_path, 
                    self.api_key, 
                    query=prompt, 
                    model_name=self.model_vlm,
                    model=self.model_vlm_pipeline,
                    max_tokens=self.max_token,
                    img_size=self.img_size,
                    temperature=self.temp,
                    top_p=self.top_p,
                    seed=self.seed,
                )
            return text 

        prompt = self.text_extractor_prompt[self.category]
        if image_path in self.img2txt_db_dict:
            text = self.img2txt_db_dict[image_path]
        else:
            if self.num_text_extraction > 1:
                text_list = []
                for _ in range(self.num_text_extraction):
                    text_list.append(text_retrival(image_path, prompt))
                print(text_list)
                test_list_str = str(text_list)
                query = "select the most frequently and probably text from the list, only give the text"
                text = txt2txt(test_list_str, api_key=self.api_key, model=self.model_llm, query=query).replace('"', '')
            else:
                text = text_retrival(image_path, prompt)
            self.img2txt_db_dict[image_path] = text
            update_json(self.img2txt_db_path, self.img2txt_db_dict)
        return text

    def text_summation(self, image_path, template="",):
        """
        Summarize the text
        """  
        if template == "":
            template = self.text_summation_prompt[self.category]
        
        text = self.text_extraction(image_path)
        
        if self.wo_summation:
            return text
        else:
            summary = txt2sum(
                input_text=text,
                few_shot_message=template,
                api_key=self.api_key,
                model=self.model_llm,
                seed=self.seed,
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
        # TODO clean this part
        if self.category in ["pushpins", "splicing_connectors", "juice_bottle"]:
            x_centroid_points = self.generate_centroid_points(x)
            score = []
            for ref_img in self.reference_img_features:
                score.append(calculate_centroidmap_disance(ref_img, x_centroid_points))
            score = np.array(score).mean()
            if score > 1: # abnormal cases e.g. no object detected
                score = 0
            else:
                if score < self.threshold:
                    score = 0
        return score 
    
    def generate_crop_img(self, x):
        img, trans_img = dino_image_transform(x)
        boxes, logit, phases = get_bbx_from_query(
            trans_img, 
            self.gdino_model, 
            box_threshold=0.35, 
            text_threshold=0.35, 
            query=self.category
        )
        patches = patch_extraction_from_box(img, boxes, patch=1)
        return patches, img
    
    def generate_score(self, query):
        score = cos_sim(query, self.reference_embedding[0])
        ab_score = cos_sim(query, self.abnormal_ref_embedding)
        output = 0 if score > ab_score else 1
        return output

    def forward(self, x):
        geo_score = 0
        if self.sliding_window:
            geo_score = self.genenerate_img_features_score(x)

        # ugly code to handle the nested list
        if isinstance(x, list):
            x = x[0]

        template = self.reference_summation[0]
        text_summation = self.text_summation(x, template=template)
        print(text_summation)
        embedding = self.text_embedding(input_text=text_summation)
        # calculate the similarity
        if self.wo_summation:
            score = 1 - cos_sim(embedding, self.reference_embedding)
        else:
            score = 1 - cos_sim(embedding, self.reference_embedding[0])
        print("anomalib score:", score, "geo score", geo_score)
        score = score + geo_score
        return score