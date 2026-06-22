import os
import random
import shutil

# Sample few shot examples from a syn data dir and create a subset in a new dir
real_shots = [8]
for shot in [100]:
    for rs in real_shots:
        print(f'doing {shot} shot, {rs} real shot')
        seed = 0
        path = f'/path/to/FakeDTD{rs}shot'
        original_train_dir = 'train'
        new_train_dir = f'train_{shot}_{seed}'
        random.seed(seed)

        classes = os.listdir(os.path.join(path, original_train_dir))
        os.mkdir(os.path.join(path, new_train_dir))
        for c in classes:
            os.mkdir(os.path.join(path, new_train_dir, c))

        for c in classes:
            imgs = os.listdir(os.path.join(path, original_train_dir, c))
            selection = random.sample(imgs, shot)
            for f in selection:
                os.symlink(os.path.join(path, original_train_dir, c, f), os.path.join(path, new_train_dir, c, f))

