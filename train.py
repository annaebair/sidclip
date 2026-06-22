import random
import numpy as np
from tqdm import tqdm

import torch
from torch import nn, autocast

import clip

from utils import get_opt_and_scheduler, parse_args, _test, create_model, \
    get_flowers_names, get_food_names
from data import get_dataloaders


def _num_classes(dataset):
    return {
        'StanfordCars': 196,
        'Flowers': 102,
        'Food': 101,
        'DTD': 47,
    }[dataset]


def _get_text(dataset, test_data, device):
    if dataset == 'StanfordCars':
        return torch.cat([clip.tokenize(f"A photo of a {c}.") for c in test_data.classes]).to(device)
    elif dataset == 'Flowers':
        flowers_names = get_flowers_names()
        return torch.cat([clip.tokenize(f"A photo of a {flowers_names[c]}, a type of flower.") for c in flowers_names.keys()]).to(device)
    elif dataset == 'Food':
        food_names = get_food_names(test_data.classes)
        return torch.cat([clip.tokenize(f"A photo of a {food_names[c]}, a type of food.") for c in test_data.classes]).to(device)
    elif dataset == 'DTD':
        return torch.cat([clip.tokenize(f"A photo of a {c} texture.") for c in test_data.classes]).to(device)


def _train(model, train_loader, loss_fn, opt, device, model_name, classifier):
    train_losses = 0.0
    train_accs = 0
    train_items = 0
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        with autocast(device_type="cuda"):
            if 'efficientnet' in model_name:
                out = model(x)
            elif 'TimEfNet' in model_name:
                out = classifier(model.encode_image(x))
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        train_losses += loss.item() * len(y)
        preds = torch.argmax(out, dim=-1)
        train_accs += torch.sum(preds == y).item()
        train_items += len(y)
    return train_losses / train_items, train_accs / train_items


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(bs, lr, weight_decay, opt_type, epochs, model_name, shot, seed, syn_shot, dataset, randaug):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = _num_classes(dataset)
    model, _ = create_model(model_name, num_classes)
    model.to(device)

    syn = syn_shot is not None
    _, test_data, train_loader, test_loader, _ = get_dataloaders(dataset, bs, randaug, shot, seed, syn, syn_shot)

    if 'TimEfNet' in model_name:
        size = 512
        student_ln = nn.LayerNorm(size, dtype=torch.float16)
        student_linear = nn.Linear(size, num_classes, dtype=torch.float16)
        text = _get_text(dataset, test_data, device)
        with torch.no_grad():
            text_features = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        student_linear.weight.data = text_features
        classifier = nn.Sequential(student_ln, student_linear).to(device)
        params = [model, classifier]
    else:
        classifier = None
        params = model

    opt, scheduler = get_opt_and_scheduler(opt_type, 'cosine_annealing', params, lr, weight_decay, epochs)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        avg_train_loss, avg_train_acc = _train(model, train_loader, loss_fn, opt, device, model_name, classifier)

        if epoch == (epochs - 1):
            model.eval()
            avg_test_loss, avg_test_acc = _test(model, test_loader, loss_fn, device, model_name, classifier)
            print(f'Epoch {epoch} | Train Acc: {round(avg_train_acc*100, 2)} | Test Acc: {round(avg_test_acc*100, 2)} | Train Loss: {round(avg_train_loss, 4)} | Test loss: {round(avg_test_loss, 4)}')
        else:
            print(f'Epoch {epoch} | Train Acc: {round(avg_train_acc*100, 2)} | Train Loss: {round(avg_train_loss, 4)}')

        scheduler.step()


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    print(args)
    train(args.batch_size, args.lr, args.wd, args.opt_type, args.epochs,
          args.model, args.shot, args.seed, args.syn_shot, args.dataset,
          args.randaug == 'true')

