"""
This is official Logic Anomaly Detection with LLM implementation by using Pytorch, Pytorch-Lightning and
anomalib as general framework
"""
from torch import Tensor, nn
from .utils import init_json, update_json 

from .text_prompt import (
    TEXT_EXTRACTOR_PROMPTS,
    TEXT_SUMMATION_PROMPTS,
    TEXT_NUM_OBJECTS_PROMPTS,
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
    patch_extraction_from_box,
    get_bbx_from_query
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
        model_embedding: str = "text-embedding-3-large",
        sliding_window: bool = False,
        gdino_cfg: str= "swint",
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.category = category
        self.sliding_window = sliding_window
        self.model_vlm = model_vlm
        self.model_embedding = model_embedding
        self.reference_embedding = None
        self.reference_summation = None
        self.img2txt_db_path = img2txt_db
        self.img2txt_db_dict: dict = init_json(img2txt_db)

        # only activate groudingdino if we applying sliding window
        self.gdino_model = load_gdino_model(cfg=gdino_cfg) if sliding_window else None
    
    def init_reference(self, reference_summation, reference_embedding):
        self.reference_summation = reference_summation
        self.reference_embedding = reference_embedding
    
    def generate_sliding_windows(self, image_path):
        if self.gdino_model is None:
            raise ValueError("Sliding window is not activated, please activate it first")
        img, transformed_img = dino_image_transform(image_path)
        boxes, _, phases = get_bbx_from_query(
            image=transformed_img, 
            model=self.gdino_model,
            query=self.category,
            box_threshold=0.2,
            text_threshold=0.32
        )
        patches = ""
        # patches = patch_extraction_from_box(img, boxes=boxes)
        number_of_objects = len(boxes)
        return {
            "number_of_objects": number_of_objects,
            "patches": patches,
        }
     
    def text_extraction(self, image_path):
        """
        Extract text from the image
        """
        def text_retrival(image_path):
            text = img2text(image_path, self.api_key, query=prompt, model=self.model_vlm)
            text = text["choices"][0]["message"]["content"]
            return text 
        prompt = TEXT_EXTRACTOR_PROMPTS[self.category]
        if image_path in self.img2txt_db_dict:
            text = self.img2txt_db_dict[image_path]
        else:
            if self.sliding_window:
                text = {}
                sliding_windows_output_dict = self.generate_sliding_windows(image_path)
                # design the text extraction for sliding windows cases
                """ use the image path as dict key 
                if the number if not correct
                {
                    "overall": "the total number of pushpins is 15,
                    "patches": [
                        list of patches description
                    ]
                 }
                } 
                """
                print(sliding_windows_output_dict)
                number_of_objects = sliding_windows_output_dict["number_of_objects"]
                text["overall"] = f"overall: total number of pushpins is {number_of_objects}"
                patch_text_list = []
                patches = sliding_windows_output_dict["patches"]
                for p in patches:
                    patch_text_list.append(text_retrival(p))
                text["patches"] = patch_text_list
                if isinstance(text, dict):
                    text = str(text)
                self.img2txt_db_dict[image_path] = text
            else:
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