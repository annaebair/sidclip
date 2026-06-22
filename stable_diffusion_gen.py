import os
import math
import torch
import argparse
import time

from diffusers import KandinskyPriorPipeline, KandinskyPipeline
from diffusers.utils import load_image, make_image_grid

from data import get_dataloaders, _syn_dir
from utils import get_flowers_names, get_food_names


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='StanfordCars')
parser.add_argument('--shot', type=int)
parser.add_argument('--syn_shot', type=int, default=50)
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
print(args.shot, args.syn_shot, args.start_idx)

start = time.time()
SHOT = args.shot  # real shot
seed = args.seed
_ , _, _, _, train_data = get_dataloaders(dataset=args.dataset, bs=32, randaug=False, shot=SHOT, seed=args.seed, syn=None, syn_shot=None)
new_data_root_dir = _syn_dir(args.dataset, SHOT, seed)
if args.dataset == 'Flowers':
    name_dict = get_flowers_names()
elif args.dataset == 'Food':
    underlying = train_data.dataset if hasattr(train_data, 'dataset') else train_data
    name_dict = get_food_names(underlying.classes)

if args.dataset == 'Flowers':
    flowers_names = get_flowers_names()
    classnames = list(flowers_names.keys())
else:
    underlying = train_data.dataset if hasattr(train_data, 'dataset') else train_data
    classnames = underlying.classes
for c in classnames:
    os.makedirs(f'{new_data_root_dir}/{c}', exist_ok=True)

prior_pipeline = KandinskyPriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
prompt = ""
pipeline = KandinskyPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
weights = [0.2, 0.4, 0.4]

setup = time.time() - start
print('setup time: ', setup)
for c in range(len(classnames)):
    if c < args.start_idx:
        continue
    if args.dataset == 'Food' or args.dataset == 'StanfordCars' or args.dataset == 'DTD':
        class_name = classnames[c]
    elif args.dataset == 'Flowers':
        class_name = c
    print('starting ', class_name)
    files = os.listdir(f'{new_data_root_dir}/{class_name}')
    if len(files) >= 300:#args.syn_shot:
        continue
    idx = c * SHOT

    images = {}
    completed = set()
    shot_to_samples_dict = {1: 1, 2: 1, 4: 6, 8: 28}
#    shot_to_start_num = {1: 50, 2:50, 4: 10, 8: 2}
    num_samples = math.ceil(args.syn_shot / shot_to_samples_dict[SHOT])
#    start_num = shot_to_start_num[SHOT]
    start_num = 0
    for s in range(SHOT):
        images[s] = train_data[idx+s][0].resize((512, 512))    

    if args.shot == 1:
        weights = [0.4, 0.6]
        if args.dataset == 'StanfordCars' or args.dataset == 'DTD':
            images_texts = [class_name, images[0]]
        else:
            images_texts= [name_dict[class_name], images[0]]
        prior_out = prior_pipeline.interpolate(images_texts, weights)
        for i in range(start_num, start_num + num_samples):
            if i < 10:
                j = f'0{i}'
            else:
                j = f'{i}'
            image = pipeline(prompt, **prior_out, height=768, width=768).images[0]
            image.save(f'{new_data_root_dir}/{class_name}/{j}1.jpg')


    else:
        for ii in images.keys():
            for jj in images.keys():
                if (ii, jj) not in completed and ii != jj:
                    start_img = time.time()
                    # images need to be PIL
                    img_ii = images[ii]
                    img_jj = images[jj]
                    if args.dataset == 'StanfordCars' or args.dataset == 'DTD':
                        images_texts = [class_name, img_ii, img_jj]
                    else:
                        images_texts = [name_dict[class_name], img_ii, img_jj]
                        print(name_dict[class_name])
                    prior_out = prior_pipeline.interpolate(images_texts, weights)
    #                image = pipeline(prompt, **prior_out, height=769, width=768).images[0]
    #                image.save(f'{new_data_root_dir}/{class_name}/0{ii}{jj}.jpg')
                    for i in range(start_num, start_num + num_samples):
                        image = pipeline(prompt, **prior_out, height=768, width=768).images[0]
                        image.save(f'{new_data_root_dir}/{class_name}/{i}{ii}{jj}.jpg')
                    
                    completed.add((ii, jj))
                    completed.add((jj, ii))


