import torch
import torch.nn as nn
import numpy as np
import logging
from torch.nn.init import xavier_uniform_


class FF(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))

def ipe(x):

    mu = x.mean(0)
    Sigma = torch.diag(torch.var(x, 0))

    exp_part = torch.exp(-0.5 * torch.diag(Sigma))
    sin_enc = torch.sin(x @ mu.unsqueeze(1)) * exp_part
    cos_enc = torch.cos(x @ mu.unsqueeze(1)) * exp_part

    encoding = torch.cat([sin_enc, cos_enc], dim=1)
    return encoding


def input_encoding(input_tensor, use_log=False):
    if use_log:
        input_tensor = torch.log(input_tensor)

    angles = input_tensor * (torch.tensor(np.pi / 2))
    encoded_vector = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)

    return encoded_vector



class HyperNetTr(nn.Module):

    def __init__(self, mainnet, embed_dim, vision_cfg, hyp_cfg, cast_dtype, _Conv_PARAM=None, init_enc_bias = 1.0, ws=False) -> None:
        super(HyperNetTr, self).__init__()

        d_model = hyp_cfg.width
        nhead = hyp_cfg.heads
        dim_ff = hyp_cfg.dim_ff
        layers = hyp_cfg.layers
        budget = hyp_cfg.bottleneck
        self.cast_dtype = cast_dtype
        activation = 'gelu'

        name_shape_dict = {name: param.shape for name, param in mainnet.named_parameters()}
        output_numels = [np.prod(shape) for shape in name_shape_dict.values()]
        logging.info("Mainnet parameters: %s", sum(output_numels))
        logging.info("activation: %s", activation)

        self.in_proj = nn.Linear(embed_dim * 2, d_model, bias=False)

        if hyp_cfg.transformer == "basic":
            self.htransformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.1,\
                                            dim_feedforward=dim_ff, batch_first=True, \
                                            activation=activation), num_layers=layers)

        elif hyp_cfg.transformer == "single":
            self.htransformer = nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True, split=self.split)
        else:
            self.htransformer = nn.Identity()

        self.image_size = mainnet.image_size
        for param in mainnet.parameters():
            param.requires_grad = hyp_cfg.mainnet_grad

        name_shape_dict = {name + '.' + _name: param_shape for name, _name, param_shape in _Conv_PARAM}

        self.name_shape_dict_keys = name_shape_dict.keys()
        self.output_shapes = name_shape_dict.values()
        self.output_numels = np.sum([np.prod(shape) for shape in self.output_shapes])

        logging.info(hyp_cfg.mode)
        logging.info("HyperNet output size: %s", self.output_numels)

        if d_model * self.output_numels > budget:
            self.bottleneck = nn.Linear(d_model, int(budget / self.output_numels))
            self.some_norm = nn.LayerNorm(int(budget / self.output_numels))
            self.out_ff = nn.Linear(int(budget / self.output_numels), self.output_numels, bias=False)
        else:
            self.bottleneck = nn.Identity()
            self.some_norm = nn.LayerNorm(d_model)
            self.out_ff = nn.Linear(d_model, self.output_numels, bias=False)

        if ws:
            self.weight_scale = nn.Parameter(torch.ones([]) * np.log(1.7924748659133911))
            self.ws = True
        else:
            self.ws = False

        name_shape_dict = {name: param.shape for name, param in self.htransformer.named_parameters()}
        output_numels = [np.prod(shape) for shape in name_shape_dict.values()]
        logging.info("HyperNet transformer size: %s", sum(output_numels))

        if isinstance(self.out_ff, nn.modules.Linear):
            lin_dim = np.prod(self.out_ff.weight.shape)
            logging.info("HyperNet linear size: %s", lin_dim)
        else:
            lin_dim = np.prod(self.out_ff.fc1.weight.shape) + np.prod(self.out_ff.fc2.weight.shape)
            logging.info("HyperNet linear size: %s", lin_dim)

        if hyp_cfg.init_transformer != "default":
            self._init_transformer()

        if hyp_cfg.init_linear != "default":
            self._init_linear()

    @staticmethod
    def convert_to_shapes(out, output_shapes):
        output_numels = [np.prod(shape) for shape in output_shapes]
        split_out = torch.split(out, output_numels)
        reshaped_out = [p.reshape(out_shape) for p, out_shape in zip(split_out, output_shapes)]

        return reshaped_out

    def _init_transformer(self):
        for p in  self.htransformer.parameters():
            if p.dim() > 1:
                 xavier_uniform_(p, nn.init.calculate_gain('relu'))


    def _init_linear(self):
        nn.init.normal_(self.out_ff.weight)
        self.out_ff.weight.data.mul_(1e-2)


    def forward(self, text_features):
        text_features = ipe(text_features)

        out = self.in_proj(text_features).float()
        out = self.htransformer(out)


        out = self.bottleneck(out)
        out = self.some_norm(out)
        params_out = self.out_ff(out.mean(0)).float()
        if self.ws:
            return params_out * self.weight_scale.exp()
        else:
            return params_out
