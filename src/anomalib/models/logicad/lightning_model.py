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
        model_vlm: str = "gpt-4o",
        model_embedding: str = "text-embedding-3-large",
        k_shot: int = 1,
    ) -> None:
        super().__init__()
        self.model: LogicadModel = LogicadModel(
            category=category,
            api_key=key_path,
            model_vlm=model_vlm,
            model_embedding=model_embedding,
        )
        self.k_shot = k_shot
        self.reference_images: list[str] = []
        self.reference_summation: list[str] = []
        self.reference_embedding: list[Tensor] = []

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
        random.shuffle(self.reference_images)
        self.reference_images = self.reference_images[: self.k_shot]
        for path in self.reference_images:
            # get list logic summation text of reference images
            text_summation = self.model.text_summation(path)
            self.reference_summation.append(text_summation)
            embedding = self.model.text_embedding(input_text=text_summation)
            self.reference_embedding.append(embedding)
        self.model.init_reference(self.reference_summation, self.reference_embedding)
        
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
        score = self.model(batch["image_path"])
        score = torch.tensor([score], device=batch['label'].device)
        batch["pred_scores"] = score
        batch["anomaly_maps"] = batch["mask"]
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
            model_embedding=hparams.model.model_embedding,
            k_shot=hparams.model.k_shot,
        )
        self.hparams: DictConfig | ListConfig  # type: ignore
        self.save_hyperparameters(hparams)
