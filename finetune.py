import os
from tqdm import tqdm

import torch
import clip
from torch import nn, autocast

from data import get_dataloaders
from utils import parse_args, get_opt, get_flowers_names, create_model


def _get_text_and_classes(dataset, test_data, device):
    if dataset == 'StanfordCars':
        text = torch.cat([clip.tokenize(f"A photo of a {c}.") for c in test_data.classes]).to(device)
        num_classes = 196
    elif dataset == 'Flowers':
        flowers_names = get_flowers_names()
        text = torch.cat([clip.tokenize(f"A photo of a {flowers_names[c]}, a type of flower.") for c in flowers_names.keys()]).to(device)
        num_classes = 102
    elif dataset == 'Food':
        text = torch.cat([clip.tokenize(f"A photo of a {c}, a type of food.") for c in test_data.classes]).to(device)
        num_classes = 101
    elif dataset == 'DTD':
        text = torch.cat([clip.tokenize(f"A photo of a {c} texture.") for c in test_data.classes]).to(device)
        num_classes = 47
    return text, num_classes


def zero_shot_eval(model_type, dataset, batch_size, randaug, shot, seed, syn_shot):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = create_model(model_type)
    model.eval()

    syn = syn_shot is not None
    _, test_data, _, test_loader, _ = get_dataloaders(dataset, batch_size, randaug, shot, seed, syn, syn_shot)
    text, _ = _get_text_and_classes(dataset, test_data, device)

    total_correct = 0
    total_items = 0
    pbar = tqdm(test_loader)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        with autocast(device_type='cuda'):
            with torch.no_grad():
                image_features = model.encode_image(images)
                text_features = model.encode_text(text)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        total_correct += torch.sum(labels == torch.argmax(similarity, dim=-1)).item()
        total_items += len(labels)
        pbar.set_postfix({'acc': total_correct / total_items})
    print(f'Zero-shot accuracy: {total_correct / total_items:.4f}')


def finetune_last_layer(bs, lr, epochs, randaug, save_name, shot, seed, model_type, dataset, opt_type, syn_shot=None):
    """Fine-tune a LayerNorm+Linear head on frozen CLIP features, initialized from text embeddings."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, size = create_model(model_type)
    model.eval()

    syn = syn_shot is not None
    _, test_data, train_loader, test_loader, _ = get_dataloaders(dataset, bs, randaug, shot, seed, syn, syn_shot)
    text, num_classes = _get_text_and_classes(dataset, test_data, device)

    ln = nn.LayerNorm(size, dtype=torch.float16)
    linear = nn.Linear(size, num_classes, dtype=torch.float16)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    linear.weight.data = text_features
    layer = nn.Sequential(ln, linear).to(device)

    optimizer = get_opt(opt_type, layer, lr, 1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs)):
        layer.train()
        train_acc, train_items, train_loss = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with autocast(device_type='cuda'):
                with torch.no_grad():
                    image_embeddings = model.encode_image(x)
            out = layer(image_embeddings)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_acc += torch.sum(y == torch.argmax(out, dim=-1)).item()
            train_items += len(y)
            train_loss += loss.item() * len(y)

        layer.eval()
        test_acc, test_items = 0, 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            with autocast(device_type='cuda'):
                with torch.no_grad():
                    out = layer(model.encode_image(x))
            test_acc += torch.sum(y == torch.argmax(out, dim=-1)).item()
            test_items += len(y)

        print(f'Epoch {epoch} | Train Acc: {train_acc/train_items:.4f} | Test Acc: {test_acc/test_items:.4f}')

    save_path = os.path.join('checkpoints', f'{save_name}.pth' if save_name else 'tmp.pth')
    torch.save(layer.state_dict(), save_path)
    print(f'Saved to {save_path}')
    return test_acc / test_items


if __name__ == '__main__':
    args = parse_args()
    print(args)
    if args.mode == 'zero_shot':
        zero_shot_eval(args.model, args.dataset, args.batch_size,
                       args.randaug == 'true', args.shot, args.seed, args.syn_shot)
    elif args.mode == 'last_layer':
        finetune_last_layer(args.batch_size, args.lr, args.epochs, args.randaug == 'true',
                            args.save_name, args.shot, args.seed, args.model, args.dataset,
                            opt_type=args.opt_type, syn_shot=args.syn_shot)

