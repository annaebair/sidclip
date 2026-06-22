import os
import random
import numpy as np
from tqdm import tqdm

import torch
from torch import nn, autocast
import torch.nn.functional as F

import clip

from data import get_dataloaders
from utils import kl_loss, dkd_loss, get_opt_and_scheduler, parse_args, _test, \
    create_model, get_flowers_names, get_food_names


def save_checkpoint(student, classifier, optimizer, epoch, loss, filename):
    state_dict = {
        'epoch': epoch,
        'model_state_dict': student.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if classifier is not None:
        state_dict['classifier_state_dict'] = classifier.state_dict()
    torch.save(state_dict, filename)


def load_checkpoint(student, classifier, optimizer, filename):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        ckpt = torch.load(filename)
        student.load_state_dict(ckpt['model_state_dict'])
        if classifier is not None:
            classifier.load_state_dict(ckpt['classifier_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print(f"Resumed from epoch {ckpt['epoch']}")
        return ckpt['epoch'], ckpt['loss']
    print(f"No checkpoint found at '{filename}'")
    return 0, None


def _num_classes(dataset):
    return {
        'StanfordCars': 196,
        'Flowers': 102,
        'Food': 101,
        'DTD': 47,
    }[dataset]


def _teacher_linear_path(dataset, teacher_type):
    paths = {
        ('StanfordCars', 'ViT-L/14'): 'checkpoints/stanfordcars_vitl14.pth',
        ('StanfordCars', 'ViT-B/32'): 'checkpoints/stanfordcars_vitb32.pth',
        ('Flowers',      'ViT-L/14'): 'checkpoints/flowers_vitl14.pth',
        ('Flowers',      'ViT-B/32'): 'checkpoints/flowers_vitb32.pth',
        ('Food',         'ViT-L/14'): 'checkpoints/food_vitl14.pth',
        ('Food',         'ViT-B/32'): 'checkpoints/food_vitb32.pth',
        ('DTD',          'ViT-L/14'): 'checkpoints/dtd_vitl14.pth',
        ('DTD',          'ViT-B/32'): 'checkpoints/dtd_vitb32.pth',
    }
    return paths[(dataset, teacher_type)]


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


def _train(model, classifier, teacher, teacher_layer, train_loader, opt, device,
           dist_method, beta, model_name, epoch, T, alpha, save_name):
    train_losses = 0.0
    train_accs = 0
    train_items = 0
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        teacher.eval()
        teacher_layer.eval()
        with autocast(device_type="cuda"):
            if 'efficientnet' in model_name:
                student_out = model(x)
            elif 'TimEfNet' in model_name:
                student_emb = model.encode_image(x)
                student_out = classifier(student_emb)
            with torch.no_grad():
                teacher_emb = teacher.encode_image(x)
                teacher_out = teacher_layer(teacher_emb).detach()

        if dist_method == 'dkd':
            loss, label_loss = dkd_loss(student_out, teacher_out, y, T, alpha, beta)
        else:
            loss, label_loss = kl_loss(student_out, teacher_out, y, T, alpha)

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_losses += label_loss.item() * len(y)
        preds = torch.argmax(student_out, dim=-1)
        train_accs += torch.sum(preds == y).item()
        train_items += len(y)

    save_checkpoint(model, classifier, opt, epoch, train_losses / train_items, save_name)
    return train_losses / train_items, train_accs / train_items


def distill(dataset, bs, lr, wd, opt_type, sched_type, epochs, model_name, shot, randaug,
            seed, dist_method, syn_shot, beta, save_name, syn_epochs, teacher_type):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = _num_classes(dataset)
    student, _ = create_model(model_name, num_classes)
    student.to(device)

    use_syn = syn_epochs > 0 and syn_shot is not None
    _, test_data, train_loader_syn, test_loader, _ = get_dataloaders(
        dataset, bs, randaug, shot, seed, use_syn, syn_shot
    )
    _, _, train_loader_no_syn, _, _ = get_dataloaders(
        dataset, bs, randaug, shot, seed, False, None
    )

    teacher, size = create_model(teacher_type)
    teacher_ln = nn.LayerNorm(size)
    teacher_linear = nn.Linear(size, num_classes)
    teacher_layer = nn.Sequential(teacher_ln, teacher_linear)
    teacher_layer.load_state_dict(torch.load(_teacher_linear_path(dataset, teacher_type)))
    teacher_layer.to(device)
    teacher.eval()
    teacher_layer.eval()

    if 'TimEfNet' in model_name:
        size = 512
        student_ln = nn.LayerNorm(size, dtype=torch.float16)
        student_linear = nn.Linear(size, num_classes, dtype=torch.float16)
        text = _get_text(dataset, test_data, device)
        with torch.no_grad():
            text_features = student.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        student_linear.weight.data = text_features
        classifier = nn.Sequential(student_ln, student_linear).to(device)
        params = [student, classifier]
    else:
        classifier = None
        params = student

    opt, scheduler = get_opt_and_scheduler(opt_type, sched_type, params, lr, wd, epochs)
    eval_loss_fn = nn.CrossEntropyLoss()

    start_epoch = 0
    if save_name != 'tmp.pth' and os.path.exists(save_name):
        start_epoch, _ = load_checkpoint(student, classifier, opt, save_name)

    for epoch in range(start_epoch, epochs):
        student.train()
        if classifier:
            classifier.train()

        epoch_train_loader = train_loader_syn if epoch < syn_epochs else train_loader_no_syn

        avg_train_loss, avg_train_acc = _train(
            student, classifier, teacher, teacher_layer, epoch_train_loader, opt, device,
            dist_method, beta, model_name, epoch, T, alpha, save_name
        )

        if epoch == (epochs - 1):
            student.eval()
            if classifier:
                classifier.eval()
            avg_test_loss, avg_test_acc = _test(
                student, test_loader, eval_loss_fn, device, model_name=model_name, classifier=classifier
            )
            print(f'Epoch {epoch} | Train Acc: {round(avg_train_acc*100, 2)} | Test Acc: {round(avg_test_acc*100, 2)} | Train Loss: {round(avg_train_loss, 4)} | Test loss: {round(avg_test_loss, 4)}')
        else:
            print(f'Epoch {epoch} | Train Acc: {round(avg_train_acc*100, 2)} | Train Loss: {round(avg_train_loss, 4)}')

        if scheduler is not None:
            scheduler.step()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    args = parse_args()
    T = args.T
    alpha = args.alpha
    sched_type = 'step_lr' if args.dataset in ('Flowers', 'DTD') else 'cosine_annealing'
    set_seed(args.seed)
    distill(
        dataset=args.dataset,
        bs=args.batch_size,
        lr=args.lr,
        wd=args.wd,
        opt_type=args.opt_type,
        sched_type=sched_type,
        epochs=args.epochs,
        model_name=args.model,
        shot=args.shot,
        randaug=args.randaug == 'true',
        seed=args.seed,
        dist_method=args.dist_method,
        syn_shot=args.syn_shot,
        beta=args.beta,
        save_name=args.save_name,
        syn_epochs=args.syn_epochs,
        teacher_type=args.teacher,
    )
