from __future__ import annotations

import logging

import torch
import random
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from anomalib.models.components import AnomalyModule

from .torch_model import LogicadModel

logger = logging.getLogger(__name__)


class Logicad(AnomalyModule):
    """
    Algorithm steps
    1. Use LLM to generate template  
    """
    def __init__(
        self,
        category: str = "breakfast_box",
        key_path: str = "",
        model_vlm: str = "gpt-4o-az",
        model_llm: str = "gpt-4o-az",
        model_embedding: str = "text-embedding-3-large",
        temp = None,
        top_p = None,
        max_token = 300,
        img_size = 128,
        img2txt_db: str=  "./dataset/loco.json",
        sliding_window: bool = False,
        croping_patch: bool = False,
        gdino_cfg: str= "swint",
        seed: int = 42,
        k_shot: int = 1,
    ) -> None:
        super().__init__()
        self.model: LogicadModel = LogicadModel(
            category=category,
            api_key=key_path,
            max_token=max_token,
            temp=temp,
            img_size=img_size,
            top_p=top_p,
            img2txt_db=img2txt_db,
            model_vlm=model_vlm,
            model_llm=model_llm,
            seed = seed,
            model_embedding=model_embedding,
            sliding_window=sliding_window,
            croping_patch=croping_patch
        )
        self.gdino_cfg = gdino_cfg
        self.sliding_window = sliding_window
        self.k_shot = k_shot
        self.reference_images: list[str] = []
        self.reference_summation: list[str] = []
        self.reference_embedding: list[Tensor] = []
        self.reference_img_features = []

    def configure_optimizers(self) -> None:
        return None

    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> None:
        """
        Store reference/good image path in reference images list
        """
        del args, kwargs
        x = batch["image_path"][0] # so far just test with batch size 1 during training
        self.reference_images.append(x)

    def on_validation_start(self) -> None:
        # select reference image
        self.reference_images = self.reference_images[: self.k_shot]
        for path in self.reference_images:
            # get list logic summation text of reference images
            text_summation = self.model.text_summation(path)
            self.reference_summation.append(text_summation)
            embedding = self.model.text_embedding(input_text=text_summation)
            self.reference_embedding.append(embedding)
            # if self.sliding_window:
            self.reference_img_features.append(self.model.generate_centroid_points(path))
        self.model.init_reference(self.reference_summation, self.reference_embedding, self.reference_img_features)
        print("Reference images: ", self.reference_summation)
        
    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Get batch of anomaly maps from input image batch.

        Args:
            batch (dict[str, str | Tensor]): Batch containing image filename,
                image, label and mask

        Returns:
            dict[str, Any]: Image filenames, test images, GT and predicted label/masks
        """
        # These variables are not used.
        del args, kwargs

        # Get anomaly maps and predicted scores from the model.
        print(batch["image_path"])
        score = self.model(batch["image_path"])
        score = torch.tensor([score], device=batch['label'].device)
        batch["pred_scores"] = score
        return batch


class LogicadLightning(Logicad):
    """PatchcoreLightning Module to train PatchCore algorithm.

    Args:
        hparams (DictConfig | ListConfig): Model params
    """

    def __init__(self, hparams) -> None:
        super().__init__(
            category=hparams.model.category,
            key_path=hparams.model.key_path,
            model_vlm=hparams.model.model_vlm,
            img2txt_db=hparams.model.img2txt_db,
            model_embedding=hparams.model.model_embedding,
            model_llm=hparams.model.model_llm,
            max_token=hparams.model.max_token,
            img_size=hparams.model.img_size,
            k_shot=hparams.model.k_shot,
            seed=hparams.model.seed,
            sliding_window=hparams.model.sliding_window,
            croping_patch=hparams.model.croping_patch,
            gdino_cfg=hparams.model.gdino_cfg,
        )
        self.hparams: DictConfig | ListConfig  # type: ignore
        self.save_hyperparameters(hparams)
