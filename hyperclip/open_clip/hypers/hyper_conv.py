import torch
import torch.nn as nn
import torch.nn.functional as F
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


# class AffineTransformation(nn.Module):
#     def __init__(self, rank, k, inc, ouc, convg, device):
#         super(AffineTransformation, self).__init__()

#         r = rank
#         self.loru = nn.Parameter(1e-2 * torch.randn(r * k, inc * k, device=device), requires_grad=True)
#         self.lorv = nn.Parameter(torch.zeros(ouc // convg * k , r * k, device=device), requires_grad=True)


#     def forward(self, x):
#         logging.info("affine")
#         x = torch.add(x, (torch.matmul(self.lorv, self.loru).view(x.shape)))
#         return x

# class LRConv2dW(nn.Module):

#     def __init__(self, conv_module:  nn.modules.Conv2d, rank=2, alpha=1., device='cuda:0') -> None:
#         super(LRConv2dW, self).__init__()

#         k = conv_module.kernel_size[0]
#         inc = conv_module.in_channels
#         ouc = conv_module.out_channels
#         convg = conv_module.groups

#         r = rank
#         self.scale = alpha / r

#         self.affine = AffineTransformation(r, k, inc, ouc, convg, device)
#         self.conv = conv_module

#     def forward(self, x):

#         W = torch.mul(self.affine(self.conv.weight), self.scale)
#         return self.conv._conv_forward(x, W, self.conv.bias)


class LRConv2d(nn.Module):

    def __init__(self, conv_module:  nn.modules.Conv2d, name, rank=2, alpha=1.) -> None:
        super(LRConv2d, self).__init__()

        k = conv_module.kernel_size[0]
        inc = conv_module.in_channels
        ouc = conv_module.out_channels
        convg = conv_module.groups

        r = rank

        self.loru = nn.Parameter(1e-2 * torch.randn(r * k, inc * k), requires_grad=False)
        self.lorv = nn.Parameter(torch.zeros(ouc // convg * k , r * k), requires_grad=False)

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

        # self.loru = self.loru - eta * loru_grad
        # self.loru = self.loru - eta * loru_grad

        W = W + (torch.matmul(self.lorv, self.loru)).view(W.shape) * self.scale
        return self.conv._conv_forward(x,  W,  b)


_Conv_PARAM = []
def conv_parameters(module, name='', rank=4, ldt=1024):
    global _Conv_PARAM

    if isinstance(module, nn.modules.Conv2d):
        module = LRConv2d(module, name, rank=rank)
        for _name, param in module.named_parameters():
            if 'lor' in _name:
                _Conv_PARAM.append((name, _name, param.shape))

    else:
        for child_name, child in module.named_children():
            full_child_name = '.'.join([name, child_name]) if name else child_name
            new_child = conv_parameters(child, full_child_name, rank=rank, ldt=ldt)
            if new_child is not child:
                module.add_module(child_name, new_child)
    return module


_Conv_bias_PARAM = []
def conv_bias_parameters(module, name='', device='cuda:0'):
    global _Conv_bias_PARAM
    res = module

    if isinstance(module, nn.modules.Conv2d):
        for _name, param in res.named_parameters():
            if 'weight' in _name:
                _Conv_bias_PARAM.append((name, _name, param.shape))
                res.requires_grad_(False)

    else:
        for child_name, child in module.named_children():
            full_child_name = '.'.join([name, child_name]) if name else child_name
            new_child = conv_bias_parameters(child, full_child_name, device=device)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res


class HyperNetTr(nn.Module):

    def __init__(self, mainnet, embed_dim, vision_cfg, hyp_cfg, cast_dtype, init_enc_bias = 1.0, ws=False) -> None:
        super(HyperNetTr, self).__init__()

        d_model = hyp_cfg.width
        nhead = hyp_cfg.heads
        dim_ff = hyp_cfg.dim_ff
        layers = hyp_cfg.layers
        budget = hyp_cfg.bottleneck
        self.cast_dtype = cast_dtype
        activation = 'relu'

        name_shape_dict = {name: param.shape for name, param in mainnet.named_parameters()}
        output_numels = [np.prod(shape) for shape in name_shape_dict.values()]
        logging.info("Mainnet parameters: %s", sum(output_numels))
        logging.info("activation: %s", activation)

        self.in_proj = nn.Linear(embed_dim * 2, d_model, bias=False) # for ipe
        # self.in_proj = FF(embed_dim * 2, d_model)

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

        #!IMPORTANT do not comment!
        #this is done before the recursive call -
        #in the recursive call, the params being updated are set to False
        for param in mainnet.parameters():
            param.requires_grad = hyp_cfg.mainnet_grad

        if hyp_cfg.mode == "ALL":
            name_shape_dict = {name: param.shape for name, param in mainnet.named_parameters()}

        elif hyp_cfg.mode == "CONV":
            conv_parameters(mainnet)
            name_shape_dict = {name + '.' + _name: param_shape for name, _name, param_shape in _Conv_PARAM}

        elif hyp_cfg.mode == "CONVBIAS":
            conv_bias_parameters(mainnet)
            name_shape_dict = {name + '.' + _name: param_shape for name, _name, param_shape in _Conv_bias_PARAM}


        self.name_shape_dict_keys = name_shape_dict.keys()
        self.output_shapes = name_shape_dict.values()
        self.output_numels = np.sum([np.prod(shape) for shape in self.output_shapes])

        logging.info(hyp_cfg.mode)
        logging.info("HyperNet output size: %s", self.output_numels)

        if d_model * self.output_numels > budget:
            self.bottleneck = nn.Linear(d_model, int(budget / self.output_numels))
            #self.bottleneck = FF(d_model, int(budget / self.output_numels))
            # self.bottleneck1 = nn.Linear(d_model, (202 * 4), False)
            # self.bottleneck2 = nn.Linear(d_model, (208 * 4), False)

            self.some_norm = nn.LayerNorm(int(budget / self.output_numels))
            self.out_ff = nn.Linear(int(budget / self.output_numels), self.output_numels, bias=False)
        else:
            self.bottleneck = nn.Identity()
            self.some_norm = nn.LayerNorm(d_model)
            self.out_ff = nn.Linear(d_model, self.output_numels, bias=False)

        if ws:
            self.weight_scale = nn.Parameter(torch.ones([]) * np.log(1.7924748659133911)) #3.01
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
        # pass
        # zeros_(self.out_ff.weight)
        nn.init.normal_(self.out_ff.weight)
        self.out_ff.weight.data.mul_(1e-2)

        # nn.init.normal_(self.out_ff.fc1.weight)
        # self.out_ff.fc1.weight.data.mul_(1e-2)

        # nn.init.normal_(self.out_ff.fc2.weight)
        # self.out_ff.fc2.weight.data.mul_(1e-2)


    def forward(self, text_features):
        # text_features = input_encoding(text_features) #baseline
        text_features = ipe(text_features) #default
        # text_features_2 = apply_rotary_emb(text_features, self.fcis)
        # text_features = torch.cat([text_features_0, text_features_1, text_features_2], dim=1)
        #text_features += self.enc_bias

        out = self.in_proj(text_features).float()
        out = self.htransformer(out)

        # out = self.htransformer(out, out, out, need_weights=False, split=self.split)
        #TODO: treat the output as low-rank factorizations - U, V; 42016= 202 * 208
        # d = out.shape[0]
        # out = self.bottleneck1(out).reshape(d, -1, 4) @ self.bottleneck2(out).reshape(d, 4, -1)

        # params_out = out.reshape(d, -1).mean(0).float()


        out = self.bottleneck(out)
        out = self.some_norm(out)
        #@todo is this is right thing to do here?
        params_out = self.out_ff(out.mean(0)).float()
        # params_out = torch.nn.functional.relu(params_out)

        if self.ws:
            return params_out * self.weight_scale.exp()
        else:
            return params_out
