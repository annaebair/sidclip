import argparse
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'hyperclip'))

import torch
from torch import nn, autocast
from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torch.nn.functional as F

import clip
import timm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--wd', type=float, default=2e-4)
    parser.add_argument('--opt_type', type=str, default='rmsprop')
    parser.add_argument('--sched_type', type=str, default='none')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--T', type=int, default=2)
    parser.add_argument('--model', type=str, default='TimEfNetb0')
    parser.add_argument('--shot', default=None)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--randaug', type=str, default='true')
    parser.add_argument('--dist_method', type=str, default='kl')
    parser.add_argument('--syn_shot', default=None)
    parser.add_argument('--mode', type=str, default='zero_shot')
    parser.add_argument('--lp', action='store_true')
    parser.add_argument('--dataset', type=str, default='StanfordCars')
    parser.add_argument('--beta', type=float, default=8.0)
    parser.add_argument('--save_name', type=str, default='tmp.pth')
    parser.add_argument('--syn_epochs', type=int, default=0)
    parser.add_argument('--teacher', type=str, default='ViT-L/14')
    args = parser.parse_args()
    return args


def get_opt(opt_type, model, lr, weight_decay):
    if isinstance(model, list):
        params = []
        for item in model:
            params.append({'params': item.parameters()})
    else:
        params = model.parameters()
    if opt_type == 'sgd':
        opt = SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif opt_type == 'adam':
        opt = Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt_type == 'rmsprop':
        opt = RMSprop(params, lr=lr, eps=1e-3, alpha=0.9, momentum=0.9, weight_decay=weight_decay)
    return opt


def get_opt_and_scheduler(opt_type, scheduler_type, model, lr, weight_decay, epochs):
    opt = get_opt(opt_type, model, lr, weight_decay)
    if scheduler_type == 'step_lr':
        scheduler = StepLR(opt, step_size=2, gamma=0.97)
    elif scheduler_type == 'cosine_annealing':
        scheduler = CosineAnnealingLR(opt, epochs)
    elif scheduler_type == 'none':
        scheduler = None
    return opt, scheduler


def _test(model, test_loader, loss_fn, device, model_name='efficientnet_b0', classifier=None):
    test_losses = 0.0
    test_accs = 0
    test_items = 0
    for x, y in test_loader:
        x, y = x.to(device), y.flatten().to(device)
        with torch.no_grad():
            with autocast(device_type="cuda"):
                if 'TimEfNet' in model_name:
                    out = model.encode_image(x)
                    out = classifier(out)
                else:
                    out = model(x)
                loss = loss_fn(out, y)
        test_losses += loss.item() * len(y)
        preds = torch.argmax(out, dim=-1)
        test_accs += torch.sum(preds == y).item()
        test_items += len(y)
    return test_losses / test_items, test_accs / test_items


def get_food_names(classnames):
    d = {}
    for c in classnames:
        d[c] = " ".join(c.split('_'))
    return d


def get_flowers_names():
    d = {"21":"fire lily","3":"canterbury bells","45":"bolero deep blue","1":"pink primrose","34":"mexican aster","27":"prince of wales feathers","7":"moon orchid","16":"globe-flower","25":"grape hyacinth","26":"corn poppy","79":"toad lily","39":"siam tulip","24":"red ginger","67":"spring crocus","35":"alpine sea holly","32":"garden phlox","10":"globe thistle","6":"tiger lily","93":"ball moss","33":"love in the mist","9":"monkshood","102":"blackberry lily","14":"spear thistle","19":"balloon flower","100":"blanket flower","13":"king protea","49":"oxeye daisy","15":"yellow iris","61":"cautleya spicata","31":"carnation","64":"silverbush","68":"bearded iris","63":"black-eyed susan","69":"windflower","62":"japanese anemone","20":"giant white arum lily","38":"great masterwort","4":"sweet pea","86":"tree mallow","101":"trumpet creeper","42":"daffodil","22":"pincushion flower","2":"hard-leaved pocket orchid","54":"sunflower","66":"osteospermum","70":"tree poppy","85":"desert-rose","99":"bromelia","87":"magnolia","5":"english marigold","92":"bee balm","28":"stemless gentian","97":"mallow","57":"gaura","40":"lenten rose","47":"marigold","59":"orange dahlia","48":"buttercup","55":"pelargonium","36":"ruby-lipped cattleya","91":"hippeastrum","29":"artichoke","71":"gazania","90":"canna lily","18":"peruvian lily","98":"mexican petunia","8":"bird of paradise","30":"sweet william","17":"purple coneflower","52":"wild pansy","84":"columbine","12":"colt's foot","11":"snapdragon","96":"camellia","23":"fritillary","50":"common dandelion","44":"poinsettia","53":"primula","72":"azalea","65":"californian poppy","80":"anthurium","76":"morning glory","37":"cape flower","56":"bishop of llandaff","60":"pink-yellow dahlia","82":"clematis","58":"geranium","75":"thorn apple","41":"barbeton daisy","95":"bougainvillea","43":"sword lily","83":"hibiscus","78":"lotus lotus","88":"cyclamen","94":"foxglove","81":"frangipani","74":"rose","89":"watercress","73":"water lily","46":"wallflower","77":"passion flower","51":"petunia"}
    new_d = {}
    for i in range(1, 103):
        new_d[i-1] = d[str(i)]
    return new_d


def kl_div(student_out, teacher_out, T):
    return T ** 2 * nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_out / T, dim=1),
        F.softmax(teacher_out / T, dim=1)
    )


def kl_loss(student_out, teacher_out, labels, T, alpha):
    teacher_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_out / T, dim=1),
        F.softmax(teacher_out / T, dim=1)
    )
    label_loss = F.cross_entropy(student_out, labels)
    loss = teacher_loss * (alpha * T ** 2) + label_loss * (1. - alpha)
    return loss, label_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def dkd_loss(student_out, teacher_out, labels, T, alpha, beta):
    gt_mask = _get_gt_mask(student_out, labels)
    other_mask = _get_other_mask(student_out, labels)
    tckd = kl_div(student_out * gt_mask, teacher_out * gt_mask, T)
    nckd = kl_div(student_out * other_mask, teacher_out * other_mask, T)
    label_loss = F.cross_entropy(student_out, labels)
    return alpha * tckd + beta * nckd, label_loss


def create_model(model_type, num_classes=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_type == 'ViT-L/14':
        model, _ = clip.load('ViT-L/14', device=device)
        size = 768
    elif model_type == 'ViT-B/32':
        model, _ = clip.load('ViT-B/32', device=device)
        size = 512
    elif model_type == 'TimEfNetb0':
        from hyperclip import open_clip as oc
        model_path = os.environ.get('TIMEFNETB0_CHECKPOINT')
        assert model_path is not None, (
            "Set the TIMEFNETB0_CHECKPOINT environment variable to the path of your TimEfNetb0 checkpoint."
        )
        model, _, _ = oc.create_model_and_transforms('TimEfNetb0', pretrained=model_path)
        model = model.to(device)
        size = 512
    elif model_type == 'TimEfNetb1':
        from hyperclip import open_clip as oc
        model_path = os.environ.get('TIMEFNETB1_CHECKPOINT')
        assert model_path is not None, (
            "Set the TIMEFNETB1_CHECKPOINT environment variable to the path of your TimEfNetb1 checkpoint."
        )
        model, _, _ = oc.create_model_and_transforms('TimEfNetb1', pretrained=model_path)
        model = model.to(device)
        size = 512
    elif model_type == 'TimEfNetb2':
        from hyperclip import open_clip as oc
        model_path = os.environ.get('TIMEFNETB2_CHECKPOINT')
        assert model_path is not None, (
            "Set the TIMEFNETB2_CHECKPOINT environment variable to the path of your TimEfNetb2 checkpoint."
        )
        model, _, _ = oc.create_model_and_transforms('TimEfNet', pretrained=model_path)
        model = model.to(device)
        size = 512
    elif model_type in ('efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2'):
        model = timm.create_model(model_type, pretrained=True, num_classes=num_classes)
        size = num_classes
    elif model_type == 'TinyViT':
        model = timm.create_model('tiny_vit_5m_224.dist_in22k_ft_in1k', pretrained=True, num_classes=num_classes)
        size = num_classes
    return model, size

