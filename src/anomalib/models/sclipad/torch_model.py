"""
This is official SCCLIPAD implementation by using Pytorch, Pytorch-Lightning and
anomalib as general framework
# This implementation is based on the following paper:
# TODO my paper link archiv
"""
import pdb
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from einops import rearrange
import copy, math


from .text_prompt import (
    TEMPLATE_LEVEL_PROMPTS,
    STATE_LEVEL_NORMAL_PROMPTS,
    STATE_LEVEL_ABNORMAL_PROMPTS,
)

from .sliding_windows import generate_sliding_windows

from .utils import (
    get_clip_encoders,
    csa_attn,
    key_smoothing,
    get_dino_attn
)


class SCLIPADModel(nn.Module):
    """
    TODO: add text prompt reasoning status on going
    """
    def __init__(
        self,
        backbone: str = "ViT-B-16",
        pretrained: str = "laion400m_e32",
        ckpt: str = None, # "./checkpoint/sd_epoch=0-step=300.ckpt",
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
        interpolate_offset=0.1,
        text_adapter: float = 0.0,
        image_adapter: float = 0.0,
        attn_logit_scale: float = 1.5,
    ) -> None:
        super().__init__()
        self.logit_scale = None
        self.tokenizer = None
        self.model = None
        self.preprocess = None
        self.clip_config = None

        self.attn_logit_scale = attn_logit_scale
        self.isCSA = isCSA
        self.clsCSA = clsCSA
        self.apply_sliding_windows = apply_sliding_windows
        self.apply_key_smoothing = apply_key_smoothing
        self.few_shot_topk = few_shot_topk
        self.cls_type_windows = cls_type_windows
        self.text_prompt_type = text_prompt_type
        self.backbone = backbone
        self.pretrained = pretrained
        self.feature_layer = feature_layer
        self.category = category
        self.k_shot = k_shot
        self.zero_shot = zero_shot
        self.interpolate_offset = interpolate_offset
        self.istimm = False
        self.get_pretrain_model_tokenizer()
        self.text_adapter = text_adapter
        self.image_adapter = image_adapter

        if text_adapter:
            embedding_dim = self.model.visual.output_dim
            self.text_linear = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, embedding_dim//4, bias=False),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(embedding_dim//4, embedding_dim, bias=False),
                torch.nn.ReLU(inplace=True),
            )
            self.text_quotient = nn.Parameter(torch.ones([]) * text_adapter)
        if image_adapter:
            embedding_dim = self.model.visual.output_dim
            self.image_linear = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, embedding_dim//4, bias=False),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(embedding_dim//4, embedding_dim, bias=False),
                torch.nn.ReLU(inplace=True),
            )
            self.image_quotient = nn.Parameter(torch.ones([]) * image_adapter)

        if ckpt:
            pretrained_encoder = torch.load(ckpt)
            # change key name and add absent key
            pretrained_encoder = {
                k[4:] if k.startswith("net.") else k: pretrained_encoder[k] for k in pretrained_encoder.keys()}
            split_params, split_layers = {}, {}
            for key in pretrained_encoder:
                if key.startswith("split_layers"):
                    layer_num = key.split(".")[1].split("_")[1]
                    if "visual_" + layer_num not in split_layers:
                        split_layers["visual_" + layer_num] = copy.deepcopy(self.model.visual.transformer.resblocks[int(layer_num)])
                if key.startswith("split_params"):
                    name = key.split(".")[1]
                    if name not in split_params:
                        split_params[name] = copy.deepcopy(self.model.visual.proj)
            if split_layers: self.split_layers = nn.ModuleDict(split_layers)
            if split_params: self.split_params = nn.ParameterDict(split_params)
            self.load_state_dict(pretrained_encoder, strict=False)
            self.eval()
        # text tower might have been altered in training, therefore build text classifier after ckpt loading
        self.text_features = torch.nn.Parameter(self.build_text_classifier(), requires_grad=False)
        self.register_buffer("ref_image_features", None)
        
        # assign sliding window size, size depends
        self.window_size = self.clip_config["vision_cfg"]["image_size"]
        self.patch_size = self.clip_config["vision_cfg"]["patch_size"]
        self.sliding_window_stride = self.window_size // 2
        self.freeze()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def get_pretrain_model_tokenizer(self):
        """
        extracted the pretrained clip model, tokenizer and hyperparameters setting
        """
        config_init = get_clip_encoders(backbone=self.backbone, pretrained=self.pretrained)
        self.model = config_init["clip_model"]
        self.preprocess = config_init["preprocess"]
        self.tokenizer = config_init["tokenizer"]
        self.logit_scale = config_init["logit_scale"]
        self.clip_config = config_init["clip_config"]
        self.istimm = config_init["timm"]
        try:
            self.isclipselfeva = config_init["clipself_eva"]
        except:
            self.isclipselfeva = False

    @torch.no_grad()
    def interpolate_pos_encoding(self, x, w, h):
        pos_embed = self.model.visual.positional_embedding
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = pos_embed.shape[0] - 1
        pos_embed = pos_embed.float()
        class_pos_embed = pos_embed[0:1, :]
        patch_pos_embed = pos_embed[1:, :]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset
        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),
            mode="bicubic",
            antialias=True,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(-1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=0).to(previous_dtype)

    @staticmethod
    def get_conv1(vit, patch_size, stride):
        # get the weight and hyperparameter from to pre-train vit (clip)
        conv1 = vit.conv1
        if stride != patch_size:
            conv_weight = vit.conv1.weight
            in_channels, out_channels, kernel = conv1.in_channels, conv1.out_channels, conv1.kernel_size
            conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel, stride, bias=False)
            # assign weights
            conv1.weight = conv_weight
        return conv1
    
    @torch.no_grad()
    def image_encoder_timm(self, x, normalize=True):
        vit = self.model.visual.trunk.eval()
        x = vit.patch_embed(x)
        x, rot_pos_embed = vit._pos_embed(x)
        resblocks = vit.blocks

        for block in resblocks[:self.feature_layer]:
            x = block(x, rot_pos_embed)

        for block in resblocks[self.feature_layer:]:
            if self.isCSA:
                csa_x = x + csa_attn(
                    block, 
                    block.norm1(x), 
                    istimm=True, 
                    attn_logit_scale=self.attn_logit_scale
                )
                csa_x = csa_x + block.mlp(block.norm2(csa_x))

                if hasattr(self, "split_layers") and "visual_11" in self.split_layers:
                    split_block = self.split_layers["visual_11"]
                    split_x = x + csa_attn(
                        split_block, 
                        split_block.norm1(x), 
                        istimm=self.istimm, 
                        attn_logit_scale=self.attn_logit_scale
                    ) 
                    split_x = split_x + split_block.mlp(split_block.norm2(split_x))

                if self.clsCSA:
                    x[0:1, ...] = csa_x[0:1, ...]
                else:
                    """only apply csa on patch_tokens"""
                    x[0:1, ...] = block(x)[0:1, ...]

                if hasattr(self, "split_layers") and "visual_11" in self.split_layers:
                    x[1:, ...] = split_x[1:, ...]
                else:
                    x[1:, ...] = csa_x[1:, ...]
            else:
                x = block(x)

        x = vit.norm(x)
        x = vit.fc_norm(x)
        x = vit.head(x)

        """ Apply MaskCLIP key smoothing on patch tokens"""
        x[:, 1:, :] = key_smoothing(x[:, 1:, :]) if key_smoothing else x[:, 1:, :]
        x = F.normalize(x, dim=-1, p=2) if normalize else x
        return {
            "cls": x[:, 0, :],
            "tokens": x[:, 1:, :]
        }

    '''
        Taken from https://github.com/wusize/CLIPSelf/blob/main/src/open_clip/eva_clip/eva_vit_model.py#L588
    '''
    @torch.no_grad()
    def image_encoder_clipselfeva(self, x, normalize=True):
        vit = self.model.visual
        bs, _, h, w = x.shape
        h = h // vit.patch_embed.patch_size[0]
        w = w // vit.patch_embed.patch_size[1]
        x = vit.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = vit.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if vit.pos_embed is not None:
            x = x + vit.rescale_positional_embedding(out_size=(h, w))
        x = vit.pos_drop(x)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        import os
        from functools import partial
        if os.getenv('RoPE') == '1':
            vit.rope.forward = partial(vit.rope.forward, patch_indices_keep=None)
            x = vit.patch_dropout(x)
        else:
            x = vit.patch_dropout(x)

        rel_pos_bias = vit.rel_pos_bias() if vit.rel_pos_bias is not None else None
        if rel_pos_bias: raise NotImplementedError
        for blk in vit.blocks[:self.feature_layer]:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        for blk in vit.blocks[self.feature_layer:]:
            if self.isCSA:
                csa_x = x + csa_attn(
                    blk, 
                    blk.norm1(x), 
                    isclipselfeva=True,
                    attn_logit_scale=self.attn_logit_scale
                )
                csa_x = csa_x + blk.mlp(blk.norm2(csa_x))
                x = csa_x
                if not self.clsCSA:
                    x[0:1, ...] = blk(x, rel_pos_bias=rel_pos_bias)[0:1, ...]
            else:
                x = blk(x, rel_pos_bias=rel_pos_bias)

        if normalize:
            x = vit.norm(x)
        x = vit.head(x)
        return {
            "cls": x[:, 0, :],
            "tokens": x[:, 1:, :]
        }

    @torch.no_grad()
    def image_encoder(self, x, normalize=True):
        """
        image encoder, last layer of vision transformer resblocks will be replaced
        by CSA (Correlative Self-Attention) model
        """
        vit = self.model.visual.eval()
        w, h = x.shape[-2:]
        x = SCLIPADModel.get_conv1(
            vit=vit, patch_size=self.patch_size, stride=self.patch_size)(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        # class embedding and positional embeddings
        x = torch.cat(
            [vit.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        if x.shape[1] != vit.positional_embedding.shape[0]:
            x = x + self.interpolate_pos_encoding(x, w, h)
        else:
            x = x + vit.positional_embedding.to(x.dtype)
        x = vit.patch_dropout(x)
        x = vit.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        resblocks = vit.transformer.resblocks

        for block in resblocks[:self.feature_layer]:
            x = block(x) 

        """ Applying Self Attention Module in the last attention block """
        for block in resblocks[self.feature_layer:]:
            if self.isCSA:
                csa_x = x + csa_attn(
                    block, 
                    block.ln_1(x), 
                    istimm=False, 
                    attn_logit_scale=self.attn_logit_scale,
                )
                csa_x = csa_x + block.mlp(block.ln_2(csa_x))

                if hasattr(self, "split_layers") and "visual_11" in self.split_layers:
                    split_block = self.split_layers["visual_11"]
                    split_x = x + csa_attn(
                        split_block, 
                        split_block.ln_1(x), 
                        istimm=self.istimm, 
                        attn_logit_scale=self.attn_logit_scale
                    ) 
                    split_x = split_x + split_block.mlp(split_block.ln_2(split_x))
                if self.clsCSA:
                    x[0:1, ...] = csa_x[0:1, ...]
                else:
                    """only apply csa on patch_tokens"""
                    x[0:1, ...] = block(x)[0:1, ...]

                if hasattr(self, "split_layers") and "visual_11" in self.split_layers:
                    x[1:, ...] = split_x[1:, ...]
                else:
                    x[1:, ...] = csa_x[1:, ...]
            else:
                x = block(x)

        x = x.permute(1, 0, 2)
        x = vit.ln_post(x) @ vit.proj

        if self.image_adapter:
            x = F.normalize(x, dim=-1, p=2)
            iq = torch.clamp(self.image_quotient, min=self.image_adapter, max=1.0)
            x[:, 1:, :] = iq * self.image_linear(x[:, 1:, :]) + (1-iq)*x[:, 1:, :]

        """ Apply MaskCLIP key smoothing on patch tokens"""
        if self.apply_key_smoothing:
            x[:, 1:, :] = key_smoothing(x[:, 1:, :])
        if normalize:
            x = F.normalize(x, dim=-1, p=2)
        return {
            "cls": x[:, 0, :],
            "tokens": x[:, 1:, :]
        }

    @torch.no_grad()
    def text_encoder(
        self,
        text: Tensor,
        normalize: bool = True
    ):
        model = self.model.eval()
        text_features = model.encode_text(text)
        if self.text_adapter:
            text_features = F.normalize(text_features, dim=-1, p=2)
            tq = torch.clamp(self.text_quotient, min=self.text_adapter, max=1.0)
            text_features = tq * self.text_linear(text_features) + (1-tq)*text_features

        if normalize:
            text_features = F.normalize(text_features, dim=-1, p=2)
        return text_features

    @torch.no_grad()
    def build_text_classifier(self):
        """
        Build zero-shot classifier with provided text-prompt
        """
        def _process_template(state_level_templates):
            text = []
            for template in TEMPLATE_LEVEL_PROMPTS:
                for state_template in state_level_templates:
                    text.append(template(state_template(self.category)))
            device = self.model.parameters().__next__().device
            texts = self.tokenizer(text).to(device=device)
            class_embeddings = self.text_encoder(texts)
            mean_class_embeddings = torch.mean(class_embeddings, dim=0, keepdim=True)
            mean_class_embeddings = F.normalize(mean_class_embeddings, dim=-1)
            return mean_class_embeddings
        normal_text_embedding = _process_template(STATE_LEVEL_NORMAL_PROMPTS)
        abnormal_text_embedding = _process_template(STATE_LEVEL_ABNORMAL_PROMPTS)
        return torch.cat([normal_text_embedding, abnormal_text_embedding], dim=0)

    def image_features_extraction(self, x):
        if self.istimm:
            img_features = self.image_encoder_timm(x)
        elif self.isclipselfeva:
            img_features = self.image_encoder_clipselfeva(x)
        else:
            img_features = self.image_encoder(x)
        # [B, D], [B, L, D]
        img_cls, img_patches = img_features["cls"], img_features["tokens"]
        return img_cls, img_patches

    def patches_logits_generation(
        self,
        img_patches,
        with_text: bool = True,
        text_logic: bool = False,
    ):
        """generating patch-wise abnormal score"""
        def img_text_similarity(img_patches):
            logit_patches = (img_patches @ self.text_features.T)
            logit_patches = (self.logit_scale * logit_patches).softmax(dim=-1)
            logit_patches = logit_patches.transpose(1, -1)
            return logit_patches
        if with_text:
            logit_patches = img_text_similarity(img_patches)
        else:
            # add language logic
            ref_patches = rearrange(self.ref_image_features, "k n l d -> (k n l) d")
            # [B, L, D] @ L^(total), D = B L L^(total)
            logit_patches = img_patches @ ref_patches.T
            if text_logic:
                # [B, L, D], [D, 2] = [B, L, 2] -> [B, L]
                logit_patches_weights = F.softmax(img_patches@ self.text_features.T, dim=-1)[..., 1:]
                logit_patches = logit_patches * logit_patches_weights
            logit_patches_val, _ = logit_patches.topk(k=self.few_shot_topk, dim=-1)
            logit_patches = logit_patches_val.mean(dim=-1)
        return logit_patches

    def cls_logits_generation(self, img_cls):
        # generate anomaly score
        # sanity check to ensure img_cls is normalized
        if len(img_cls.shape) == 2:
            dim = 1
        elif len(img_cls.shape) == 3:
            dim = 2
        else:
            raise ValueError("The shape of img_cls can only be 2 or 3")
        img_cls = F.normalize(img_cls, dim=dim, p=2)
        # [B, D] @ [D, 2] = [B, 2] or # [B,N, D] @ [D, 2] = [B, N, 2]
        logits_image = 100 * img_cls @ self.text_features.T[:, :2]
        logits_image = (logits_image).softmax(dim=-1)
        return logits_image

    def sim2mask(self, x, simmap):
        mask = F.interpolate(simmap, x.shape[2:], mode='bilinear')
        return mask

    def generate_mask_weights(self, x, window_size=None):
        """generate the sliding windows and mask if the input image size is
        not same as the CLIP encoder default size,
        e.g. if image has size of 240x480, the image will be cut and divided
        into few overlapped small windows
        """
        if not window_size:
            window_size = self.window_size

        output = generate_sliding_windows(
            x=x,
            patch_size=self.patch_size,
            stride=self.sliding_window_stride,
            window_size=window_size
        )
        mask, weights = output["mask"], output["weights"]
        return mask, weights
    
    def features_extraction_windows(
        self, 
        x, 
        batch_size, 
        stride=112, 
        window=224, 
        rescale=True,
    ):
        assert len(x.shape) == 4
        x_windows = x.unfold(2, window, stride).unfold(3, window, stride)
        x_windows = rearrange(x_windows,  "b c nw nh w h -> (b nw nh) c w h") # [B, Num_Win, C, W, H]

        # extract image features from sliding windows
        list_img_cls, list_img_patches = self.image_features_extraction(x_windows)

        # rescale image to the original image encoder size to get cls score
        if rescale:
            if x.shape[2] != self.window_size:
                x = F.interpolate(x, size=(self.window_size, self.window_size), mode="bilinear")
            if self.istimm:
                cls = self.image_encoder_timm(x)["cls"]
            elif self.isclipselfeva:
                cls = self.image_encoder_clipselfeva(x)["cls"]
            else:
                cls = self.image_encoder(x)["cls"]
            list_img_cls = cls[:, None, ...]
        list_img_patches = rearrange(list_img_patches, "(b n) l d -> b n l d", b=batch_size)
        return list_img_cls, list_img_patches
    
    def _generate_sliding_simmap_ascore(self, x):
        """
        TODO this part of the code need more explanation
        """
        # get the sliding windows from the query image using unfold
        global window_logits_few_shot, weighted_window_logit_few_shot
        batch_size = x.shape[0]

        window_size = self.window_size if self.apply_sliding_windows else x.shape[-1]
        # [B, N, D], [B, N, L, D]
        img_cls, img_patches = self.features_extraction_windows(
            x, batch_size=batch_size, stride=self.window_size // 2, window=window_size)
        # [number of sliding windows, L^hat] L^hat for the sliding window
        masks, weights = self.generate_mask_weights(x, window_size=window_size)

        # placeholder for saving all the anomaly maps
        weighted_window_logit = torch.zeros((batch_size, weights.shape[0]), device=x.device)
        if not self.zero_shot:
            weighted_window_logit_few_shot = torch.zeros((batch_size, weights.shape[0]), device=x.device)

        for bidx in range(batch_size):
            # [num_sliding_window, L^hat, D]
            window = img_patches[bidx]
            # [num_sliding_window, L^hat, 2]
            window_logits = self.patches_logits_generation(window).transpose(1, -1)
            window_logits = window_logits[..., 1]  # select abnormal score [num_slide_win, L^hat]

            if not self.zero_shot:
                window_logits_few_shot = self.patches_logits_generation(window, with_text=False)

            for mask, logit in zip(masks, window_logits):
                weighted_window_logit[bidx, mask] += logit

            if not self.zero_shot:
                for mask, logit in zip(masks, window_logits_few_shot):
                    weighted_window_logit_few_shot[bidx, mask] += 0.5 * (1 - logit)

        # get the smooth weighted with harmonic averaging
        weighted_window_logit = weighted_window_logit / weights  # [B, L]

        if not self.zero_shot:
            weighted_window_logit_few_shot = weighted_window_logit_few_shot / weights  # [B, L]
            weighted_window_logit = weighted_window_logit_few_shot

        # SO Far the best opinion is max
        # TODO wrappe the following part in a static function
        anomaly_score = self.cls_logits_generation(img_cls)[..., 1]
        if len(anomaly_score.shape) == 2:
            if self.cls_type_windows == "max":
                anomaly_score = anomaly_score.max(dim=-1)[0]
            elif self.cls_type_windows == "mean":
                anomaly_score = anomaly_score.mean(dim=-1)
            elif self.cls_type_windows == "topk":
                anomaly_score = anomaly_score.topk(3, dim=-1, largest=True)[0].mean(dim=1)
            elif self.cls_type_windows == "logit":
                anomaly_score = weighted_window_logit.topk(3, dim=1, largest=True)[0].mean(dim=1)
            elif self.cls_type_windows == "ensemble":
                # combined method max with logit
                anomaly_score = anomaly_score.max(dim=-1)[0] + \
                    weighted_window_logit.topk(3, dim=1, largest=True)[0].mean(dim=1)
            else:
                raise NotImplementedError("Unknown cls window type")

        width = int(weighted_window_logit.shape[-1] ** 0.5)
        simmap = weighted_window_logit.reshape(batch_size, width, width)

        if not self.zero_shot:
            simmap_reshape = rearrange(simmap, "b h w -> b (h w)")
            max_patch_logit = simmap_reshape.topk(self.few_shot_topk, dim=-1)[0].mean(dim=-1, keepdim=True)
            anomaly_score = max_patch_logit.squeeze()

        return simmap[:, None, ...],  anomaly_score

    def build_image_classifier(self, images):
        """
        extract image features vectors from the reference images and build memory bank for
        few-shot anomaly detection
        """
        batch_size, width = images.shape[0], images.shape[-1]
        window_size = self.window_size if self.apply_sliding_windows else width
        # [batch_size, num_of_sliding_windows, L, D]
        _, img_features_memory_bank = self.features_extraction_windows(
            images, batch_size=batch_size, window=window_size, stride=self.window_size // 2)
        if self.ref_image_features:
            self.ref_image_features = torch.cat((self.ref_image_features, img_features_memory_bank), dim=0)
        else:
            self.ref_image_features = img_features_memory_bank

    def forward(self, x):
        anomaly_map, anomaly_score = self._generate_sliding_simmap_ascore(x)
        anomaly_map = self.sim2mask(x, anomaly_map)
        return {
            "anomaly_map": anomaly_map,
            "anomaly_score": anomaly_score
        }
