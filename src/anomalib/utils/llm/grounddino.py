import torch
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.config import GroundingDINO_SwinT_OGC as gdino_swint_cfg
from groundingdino.config import GroundingDINO_SwinB_cfg as gdino_swinb_cfg
from omegaconf import OmegaConf


# help functions
def dict_from_class(cls):
    return dict((key, value) for (key, value) in cls.__dict__.items() if not key.startswith("__"))


def load_path(path):
    path_dict = OmegaConf.load(path)
    return path_dict


def load_gdino_model(ckpt_path=None, cfg="swint", device="cpu", type="groundingdino"):
    if type == "groundingdino":
        # convert the config module to dict
        if cfg is None:
            raise ValueError("cfg is required for groundingdino model, please select either swintb or swint")
        else:
            if cfg == "swint":
                cfg_cls = gdino_swint_cfg
            elif cfg == "swinb":
                cfg_cls = gdino_swinb_cfg
            else:
                raise ValueError("cfg should be either swint or swinb")
        cfg_dict = SLConfig(dict_from_class(cfg_cls))
        model = build_model(cfg_dict)
        if ckpt_path is None:
            if cfg == "swint":
                ckpt_path = "./datasets/GroundingDino/groundingdino_swint_ogc.pth"
            elif cfg == "swinb":
                ckpt_path = "./datasets/GroundingDino/groundingdino_swinb.pth"
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        _ = model.eval()
    else:
        raise ValueError("model type not supported")
    return model
