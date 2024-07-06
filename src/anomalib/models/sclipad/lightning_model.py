from __future__ import annotations

import logging

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from anomalib.models.components import AnomalyModule
from anomalib.models.sclipad.torch_model import SCLIPADModel

logger = logging.getLogger(__name__)


class Sclipad(AnomalyModule):
    """
    SCLIPADLightning Module to train SCLIP based Anomaly Detection algorithm
    """

    def __init__(
        self,
        backbone: str = "ViT-B-16",
        pretrained: str = "laion400m_e32",
        ckpt: str = "",
        category: str = "wood",
        k_shot: int = 4,
        zero_shot: bool = True,
        feature_layer: int = -1,
        cls_type_windows: str = "max",
        text_prompt_type: str = "standard",
        few_shot_topk: int = 1,
        apply_key_smoothing: bool = True,
        isCSA: bool = True,
        clsCSA: bool = False,
        apply_sliding_windows: bool = True,
        text_adapter: float = 0.0,
        image_adapter: float = 0.0,
        attn_logit_scale: float = 100,
    ) -> None:
        super().__init__()

        self.model: SCLIPADModel = SCLIPADModel(
            backbone=backbone,
            pretrained=pretrained,
            ckpt=ckpt,
            category=category,
            k_shot=k_shot,
            zero_shot=zero_shot,
            feature_layer=feature_layer,
            cls_type_windows=cls_type_windows,
            text_prompt_type=text_prompt_type,
            few_shot_topk=few_shot_topk,
            apply_key_smoothing=apply_key_smoothing,
            isCSA=isCSA,
            clsCSA=clsCSA,
            apply_sliding_windows=apply_sliding_windows,
            text_adapter=text_adapter,
            image_adapter=image_adapter,
            attn_logit_scale=attn_logit_scale,
        )

        self.k_shot = k_shot
        self.reference_images: list[Tensor] = []

    def configure_optimizers(self) -> None:
        """Configure optimizers.

        Returns:
            None: Do not set optimizers by returning None.
        """
        return None

    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> None:
        """Generate feature embedding of the batch.

        Args:
            batch (dict[str, str | Tensor]): Batch containing image filename, image, label and mask

        Returns:
            dict[str, np.ndarray]: Embedding Vector
        """
        del args, kwargs
        x = batch["image"]
        self.reference_images.append(x)

    def on_validation_start(self) -> None:
        ref_img = torch.cat(self.reference_images, dim=0)
        ref_img = ref_img[torch.randperm(ref_img.shape[0])[: self.k_shot]]
        self.model.build_image_classifier(ref_img)

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
        output = self.model(batch["image"])

        # Add anomaly maps and predicted scores to the batch.
        batch["anomaly_maps"] = output["anomaly_map"]
        batch["pred_scores"] = output["anomaly_score"]

        return batch


class SclipadLightning(Sclipad):
    """PatchcoreLightning Module to train PatchCore algorithm.

    Args:
        hparams (DictConfig | ListConfig): Model params
    """

    def __init__(self, hparams) -> None:
        super().__init__(
            backbone=hparams.model.backbone,
            pretrained=hparams.model.pretrained,
            ckpt=hparams.model.ckpt,
            category=hparams.model.category,
            k_shot=hparams.model.k_shot,
            zero_shot=hparams.model.zero_shot,
            feature_layer=hparams.model.feature_layer,
            cls_type_windows=hparams.model.cls_type_windows,
            text_prompt_type=hparams.model.text_prompt_type,
            few_shot_topk=hparams.model.few_shot_topk,
            apply_key_smoothing=hparams.model.apply_key_smoothing,
            isCSA=hparams.model.isCSA,
            clsCSA=hparams.model.clsCSA,
            apply_sliding_windows=hparams.model.apply_sliding_windows,
            text_adapter=hparams.model.text_adapter,
            image_adapter=hparams.model.image_adapter,
            attn_logit_scale=hparams.model.attn_logit_scale,
        )
        self.hparams: DictConfig | ListConfig  # type: ignore
        self.save_hyperparameters(hparams)
