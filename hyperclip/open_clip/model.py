""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import dataclasses
from dataclasses import dataclass
import logging
import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .hf_model import HFTextEncoder
from .modified_resnet import ModifiedResNet
from .timm_model import TimmModel
from .transformer import LayerNormFp32, LayerNorm, QuickGELU, Attention, VisionTransformer, TextTransformer, ByteTransformer, ByteVisionTransformer
from .utils import to_2tuple
from functools import partial

#from .modified_mobilenet import build_vision_mobilenet, MobileNet_V3_CLIP_Weights
#from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights
from .hyper_model import  HyperNetTr
# from .mamba_model import MambaHiddenModel, MambaConfig

#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter('log/weight')

@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    input_patchnorm: bool = False  # whether to use dual patchnorm - would only apply the input layernorm on each patch, as post-layernorm already exist in original clip vit design
    global_average_pool: bool = False  # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer
    n_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    output_tokens: bool = False

    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth
    hyper: bool = False
    mobnet: bool = False
    mobilenet_size: str = "mobilenet_v3_small"
    mobnet_pretrained: str = 'clip'


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    tokenizer_kwargs: Optional[dict] = None
    hf_model_pretrained: bool = True
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False
    no_causal_mask: bool = False
    proj_bias: bool = False
    pool_type: str = 'argmax'
    norm_kwargs: dict = None

@dataclass
class HyperCfg:
    clipinit: bool = False
    clippath: str = "datacomp-scale-medium-seed927/checkpoints/epoch_8.pt"
    mode: str = "BN" #BN/CONV/ALL,CONVLINEAR,LINEAR,CONVBIAS
    rank: int = 4
    width: int = 512
    heads: int = 8
    layers: int = 12
    dim_ff: int = 2048
    lora_dim_thresh: int = 1024
    bottleneck: int = 12000000
    mainnet_grad: bool = True
    transformer: str = "basic" #basic/clip
    init_transformer: str = "default"
    init_linear: str = "default" #default/zeros
    avg_pool: bool = False
    posenc: bool = False
    norm_first: bool = False
    act: str = "gelu"

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ('bf16', 'pure_bf16'):
        input_dtype = torch.bfloat16
    elif precision in ('fp16', 'pure_fp16'):
        input_dtype = torch.float16
    return input_dtype


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):

    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    # elif vision_cfg.mobnet:
    #     if vision_cfg.mobnet_pretrained == 'clip':
    #         weights = MobileNet_V3_CLIP_Weights.SMALL
    #     elif vision_cfg.mobnet_pretrained == 'imagenet':
    #         weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 #@todo: use weights from pretrained datacomp mobnet
    #     else:
    #         weights = None
    #     visual = build_vision_mobilenet(embed_dim, vision_cfg.image_size, vision_cfg.mobilenet_size, weights=weights)

    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            input_patchnorm=vision_cfg.input_patchnorm,
            global_average_pool=vision_cfg.global_average_pool,
            attentional_pool=vision_cfg.attentional_pool,
            n_queries=vision_cfg.n_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    return visual


def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj=text_cfg.proj,
            pooler_type=text_cfg.pooler_type,
            pretrained=text_cfg.hf_model_pretrained,
            output_tokens=text_cfg.output_tokens,
        )
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            output_tokens=text_cfg.output_tokens,
            pad_id=text_cfg.pad_id,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return text



class HyperCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            hyper_cfg: HyperCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
    ):
        super().__init__()
        self.output_dict = output_dict

        mainnet = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

        if isinstance(hyper_cfg, dict):
            hyper_cfg = HyperCfg(**hyper_cfg)

        if isinstance(text_cfg, dict):
            text_cfg = CLIPTextCfg(**text_cfg)

        self.hyp_cfg = hyper_cfg

        self.mainnet = mainnet
        # conv_parameters(self.mainnet)
        # self.visual = HyperNetTr(mainnet, embed_dim, vision_cfg, hyper_cfg, cast_dtype, _Conv_PARAM=_Conv_PARAM)
        self.visual = HyperNetTr(mainnet, embed_dim, text_cfg, hyper_cfg, cast_dtype)


    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, textfeatures, normalize: bool = False, bypass=False):

        params_out = self.visual(textfeatures)
        params_out_reshaped = self.visual.convert_to_shapes(params_out, self.visual.output_shapes)

        if self.hyp_cfg.mode == 'LN' or self.hyp_cfg.mode == 'BN':
            param_dict_b = {name : p for name, p in zip(self.visual.name_shape_dict_keys, params_out_reshaped) if 'bias' in name}
            param_dict_w = {name : p.exp() for name, p in zip(self.visual.name_shape_dict_keys, params_out_reshaped) if 'weight' in name}

            param_dict = param_dict_b
            param_dict.update(param_dict_w)

        else:
            param_dict = {name : p for name, p in zip(self.visual.name_shape_dict_keys, params_out_reshaped)}

        if bypass:
            features = self.mainnet(image)
        else:
            # analyze(param_dict)
            features = torch.func.functional_call(self.mainnet, param_dict, image, strict=False)

        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        x = self.transformer(x, attn_mask=self.attn_mask)

        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

        # x_ = x @ self.text_projection
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        # return (F.normalize(x, dim=-1), F.normalize(x_, dim=-1)) if normalize else (x, x_)

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None
    ):
        # text_features, full = self.encode_text(text, normalize=True) if text is not None else None
        # image_features = self.encode_image(image, full, normalize=True) if image is not None else None

        text_features = self.encode_text(text, normalize=True) if text is not None else None
        image_features = self.encode_image(image, text_features, normalize=True) if image is not None else None

        if self.output_dict:
            output_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                output_dict['logit_bias'] = self.logit_bias
            return output_dict
        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias

        return image_features, text_features, self.logit_scale.exp()

import os
@torch.no_grad()
def update_params_out(params_out_, path="", ap=""):
    model = 'efnetbn-hclip' + ap
    file_path = os.path.join(path, f"params_out_{model}.npy")
    params_out_ = np.expand_dims(params_out_, 0)
    if os.path.exists(file_path):
        existing_data = np.load(file_path, allow_pickle=True)
        updated_data = np.concatenate((existing_data, params_out_), axis=0)
        np.save(file_path, updated_data)
    else:
        np.save(file_path, params_out_)


global_ct = 0;
def analyze(w_dict: dict):
    global global_ct;
    params = []
    params_grad = []

    for n, p in w_dict.items():
        if p.grad is not None:
            params_grad.append(p.grad.cpu().numpy().flatten())

        params.append(p.detach().cpu().numpy().flatten())

    update_params_out(np.concatenate(params))

    if len(params_grad) > 0:
        update_params_out(np.concatenate(params_grad), ap="-grad")


_PARAM = []
def norm_parameters(module, name='', running_stats=True):
    global _PARAM
    res = module

    if isinstance(module, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm)):
        module.track_running_stats = running_stats

        if module.affine:
            for _name, param in module.named_parameters():
                _PARAM.append((name, _name, param))

            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
    else:

        for child_name, child in module.named_children():
            full_child_name = '.'.join([name, child_name]) if name else child_name
            new_child = norm_parameters(child, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res



class LRConv2d(nn.Module):

    def __init__(self, conv_module:  nn.modules.Conv2d, name, rank=2, alpha=1.) -> None:
        super(LRConv2d, self).__init__()

        k = conv_module.kernel_size[0]
        inc = conv_module.in_channels
        ouc = conv_module.out_channels
        convg = conv_module.groups

        r = rank

        self.loru0 = nn.Parameter(1e-2 * torch.randn(r * k, inc * k), requires_grad=False)
        self.lorv0 = nn.Parameter(torch.randn(ouc // convg * k , r * k), requires_grad=False)

        torch.nn.init.kaiming_uniform_(self.loru0, a=math.sqrt(5))
        self.loru = nn.Parameter(self.loru0.clone(), requires_grad=True)

        torch.nn.init.kaiming_normal_(self.lorv0, nonlinearity='relu')
        self.lorv = nn.Parameter(self.lorv0.clone(), requires_grad=True)

        # self.loru = nn.Parameter(1e-2 * torch.randn(r * k, inc * k), requires_grad=True)
        # self.lorv = nn.Parameter(torch.zeros(ouc // convg * k , r * k), requires_grad=True)

        self.register_parameter(name.replace(".", "_") + '_loru', self.loru)
        self.register_parameter(name.replace(".", "_") + '_lorv', self.lorv)

        r = rank

        self.conv = conv_module
        self.scale = alpha / r

    def forward(self, x):
        W = self.conv.weight
        if self.conv.bias is not None:
            b = self.conv.bias
        else:
            b = None

        # W = W + (torch.matmul(self.lorv, self.loru)).view(W.shape) * self.scale
        W = W - (torch.matmul(self.lorv0, self.loru0)).view(W.shape) + (torch.matmul(self.lorv, self.loru)).view(W.shape) * self.scale

        return self.conv._conv_forward(x,  W,  b)


_Conv_PARAM = []
def conv_parameters(module, name='', rank=2, ldt=1024):
    global _Conv_PARAM

    if isinstance(module, nn.modules.Conv2d):
        module = LRConv2d(module, name, rank=rank)
        # for _name, param in module.named_parameters():
        #     if 'lorv' in _name:
        #         param.requires_grad_(False)
        #         _Conv_PARAM.append((name, _name, param.shape))

    else:
        for child_name, child in module.named_children():
            full_child_name = '.'.join([name, child_name]) if name else child_name
            new_child = conv_parameters(child, full_child_name, rank=rank, ldt=ldt)
            if new_child is not child:
                module.add_module(child_name, new_child)
    return module



class CLIP_(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
            init_logit_scale: float = np.log(10),
            init_logit_bias: Optional[float] = -10,
    ):
        super().__init__()
        self.output_dict = output_dict

        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        vocab_size = 257
        context_length = 24000

        if isinstance(text_cfg, dict):
            text_cfg = CLIPTextCfg(**text_cfg)
            vision_cfg = CLIPVisionCfg(**vision_cfg)

        # self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        # self.visual = ByteTransformer(
        #     vision_cfg.image_size,
        #     context_length=context_length,
        #     vocab_size=vocab_size,
        #     width=text_cfg.width,
        #     heads=text_cfg.heads,
        #     layers=text_cfg.layers,
        #     ls_init_value=text_cfg.ls_init_value,
        #     output_dim=embed_dim,
        #     embed_cls=True,
        #     output_tokens=text_cfg.output_tokens,
        #     pad_id=256,
        #     act_layer=act_layer,
        #     norm_layer=norm_layer,
        # )

        vision_heads = vision_cfg.width // vision_cfg.head_width
        self.visual = ByteVisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            input_patchnorm=vision_cfg.input_patchnorm,
            global_average_pool=vision_cfg.global_average_pool,
            attentional_pool=vision_cfg.attentional_pool,
            n_queries=vision_cfg.n_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        # import pdb; pdb.set_trace()

        name_shape_dict = {name: param.shape for name, param in self.visual.named_parameters()}
        output_numels = [np.prod(shape) for shape in name_shape_dict.values()]
        logging.info("byte transformer size: %s", sum(output_numels))

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)


    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        # self.mainnet.set_grad_checkpointing(enable)
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)

        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None
    ):


        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None
        if self.output_dict:
            output_dict =  {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                output_dict['logit_bias'] = self.logit_bias
            return output_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias

        return image_features, text_features, self.logit_scale.exp()



class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
            init_logit_scale: float = np.log(10),
            init_logit_bias: Optional[float] = -10,
    ):
        super().__init__()
        self.output_dict = output_dict
        # self.mainnet = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        # conv_parameters(self.visual)


        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        # self.mainnet.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)


    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        # self.mainnet.set_grad_checkpointing(enable)
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: bool = False):
        # features = self.mainnet(image)

        # param_dict = {name + name_ : p for name, name_, p in _Conv_PARAM}
        # analyze(param_dict)
        features = self.visual(image)

        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None
    ):

        # ///comment
        # norm_parameters(self.visual)
        # param_dict = {name + name_ : p for name, name_, p in _PARAM}
        # analyze(param_dict)
        # ///comment

        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None
        if self.output_dict:
            output_dict =  {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                output_dict['logit_bias'] = self.logit_bias
            return output_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias

        return image_features, text_features, self.logit_scale.exp()



# class MambaCLIP(nn.Module):
#     output_dict: torch.jit.Final[bool]

#     def __init__(
#             self,
#             embed_dim: int,
#             vision_cfg: CLIPVisionCfg,
#             text_cfg: MambaConfig,
#             quick_gelu: bool = False,
#             cast_dtype: Optional[torch.dtype] = None,
#             output_dict: bool = False,
#             init_logit_scale: float = np.log(1 / 0.07),
#             init_logit_bias: Optional[float] = None,
#     ):
#         super().__init__()
#         self.output_dict = output_dict
#         self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
#         if isinstance(text_cfg, dict):
#             text_cfg = MambaConfig(**text_cfg)

#         text = MambaHiddenModel(text_cfg, embed_dim)

#         self.backbone = text.backbone
#         self.text_projection = text.text_projection
#         self.ln_final = text.ln_final
#         self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
#         if init_logit_bias is not None:
#             self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
#         else:
#             self.logit_bias = None
#         self.mamba = text

#     def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
#         self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)


#     @torch.jit.ignore
#     def set_grad_checkpointing(self, enable=True):
#         self.visual.set_grad_checkpointing(enable)
#         self.backbone.grad_checkpointing = enable

#     def encode_image(self, image, normalize: bool = False):
#         features = self.visual(image)
#         return F.normalize(features, dim=-1) if normalize else features

#     def encode_text(self, text, normalize: bool = False):
#         cast_dtype = self.backbone.get_cast_dtype()
#         # import pdb; pdb.set_trace()
#         # take features from the eot embedding (eot_token is the highest number in each sequence)
#         x = self.mamba(text, num_last_tokens=1).to(cast_dtype) #[batch_size, n_ctx(1), backbone.width]
#         x = x.squeeze(1)
#         x = self.ln_final(x)
#         x = x @ self.text_projection
#         return F.normalize(x, dim=-1) if normalize else x

#     def forward(
#             self,
#             image: Optional[torch.Tensor] = None,
#             text: Optional[torch.Tensor] = None,
#     ):
#         image_features = self.encode_image(image, normalize=True) if image is not None else None
#         text_features = self.encode_text(text, normalize=True) if text is not None else None
#         if self.output_dict:
#             output_dict =  {
#                 "image_features": image_features,
#                 "text_features": text_features,
#                 "logit_scale": self.logit_scale.exp()
#             }
#             if self.logit_bias is not None:
#                 output_dict['logit_bias'] = self.logit_bias
#             return output_dict

#         if self.logit_bias is not None:
#             return image_features, text_features, self.logit_scale.exp(), self.logit_bias

#         return image_features, text_features, self.logit_scale.exp()


class CustomTextCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = -10.,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None
        if self.output_dict:
            output_dict =  {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                output_dict['logit_bias'] = self.logit_bias
            return output_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias

        return image_features, text_features, self.logit_scale.exp()


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""


    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        if isinstance(l, (CLIP, TextTransformer)):
            # convert text nn.Parameter projections
            attr = getattr(l, "text_projection", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

        if isinstance(l, VisionTransformer):
            # convert vision nn.Parameter projections
            attr = getattr(l, "proj", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat


# used to maintain checkpoint compatibility
def convert_to_custom_text_state_dict(state_dict: dict):
    if 'text_projection' in state_dict:
        # old format state_dict, move text tower -> .text
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in (
                'text_projection',
                'positional_embedding',
                'token_embedding',
                'transformer',
                'ln_final',
            )):
                k = 'text.' + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict


def build_model_from_openai_state_dict(
        state_dict: dict,
        quick_gelu=True,
        cast_dtype=torch.float16,
):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        cast_dtype=cast_dtype,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)

    convert_weights_to_fp16(model)  # OpenAI state dicts are partially converted to float16
    model.load_state_dict(state_dict)
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device('cpu')):
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones((batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros((batch_size, model.context_length), dtype=torch.int, device=device)
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,)
        ))
    model.visual.image_size = image_size
    return model


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', antialias: bool = True):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    # new_seq_len = 1500 + 1
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed
