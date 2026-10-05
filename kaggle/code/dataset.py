from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T

NUM_CLASSES = 9
CLASS_NAMES = ['Chinee Apple', 'Lantana', 'Parkinsonia', 'Parthenium',
               'Prickly Acacia', 'Rubber Vine', 'Siam Weed', 'Snake Weed', 'Negatives']
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def discover(input_root='/kaggle/input', images_dir=None, labels_dir=None):
    root = Path(input_root)
    if labels_dir is None:
        candidates = sorted({p.parent for p in root.rglob('train_subset0.csv')})
        if len(candidates) != 1:
            raise ValueError(f'Expected one labels directory, found {candidates}. Set LABELS_DIR explicitly.')
        labels_dir = candidates[0]
    train, _, _ = load_split(labels_dir)
    if not (Path(labels_dir)/'labels.csv').is_file():
        raise FileNotFoundError('labels.csv is required for official class names and final evaluation')
    if images_dir is None:
        candidates = sorted({p.parent for p in root.rglob(str(train.iloc[0].Filename))})
        if len(candidates) != 1:
            raise ValueError(f'Expected one images directory, found {candidates}. Set IMAGES_DIR explicitly.')
        images_dir = candidates[0]
    return str(images_dir), str(labels_dir)

def load_split(labels_dir, fold=0):
    return tuple(pd.read_csv(Path(labels_dir) / f'{part}_subset{fold}.csv')
                 for part in ('train', 'val', 'test'))

def check_split(train_df, val_df, test_df, images_dir):
    frames = dict(zip(('train', 'val', 'test'), (train_df, val_df, test_df)))
    sets, result = {}, {'n': {}, 'per_class': {}, 'overlap': {}}
    for part, df in frames.items():
        # Original subset CSVs may contain only Filename and Label. Species is
        # descriptive metadata, not needed to load images or verify splits.
        required = {'Filename', 'Label'}
        if not required.issubset(df.columns):
            raise ValueError(f'{part}: missing columns {sorted(required-set(df.columns))}; '
                             f'actual columns: {df.columns.tolist()}')
        if df.Filename.duplicated().any() or df[['Filename','Label']].isna().any().any():
            raise ValueError(f'{part}: duplicate filenames or missing values')
        if not df.Label.isin(range(9)).all():
            raise ValueError(f'{part}: labels must be integers 0..8')
        sets[part] = set(df.Filename)
        missing = [f for f in sets[part] if not (Path(images_dir) / f).is_file()]
        if missing:
            raise FileNotFoundError(f'{part}: {len(missing)} missing images, e.g. {missing[:3]}')
        result['n'][part] = len(df)
        result['per_class'][part] = df.Label.value_counts().reindex(range(9), fill_value=0).tolist()
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        result['overlap'][f'{a}/{b}'] = len(sets[a] & sets[b])
        if sets[a] & sets[b]:
            raise ValueError(f'Overlapping {a}/{b}')
    if len(set.union(*sets.values())) != 17509:
        raise ValueError('Split union must contain exactly 17509 images')
    for part, ratio in [('train', .6), ('val', .2), ('test', .2)]:
        if abs(len(frames[part]) / 17509 - ratio) > .01:
            raise ValueError(f'{part}: split ratio differs by more than 1 percentage point')
    return result

def build_transforms(train, img_size=224, aug='basic', mean=IMAGENET_MEAN, std=IMAGENET_STD):
    if train:
        ops = [T.RandomResizedCrop(img_size), T.RandomHorizontalFlip()]
        extras = {'basic': [], 'color': [T.ColorJitter(.2, .2, .2, .05)],
                  'trivial': [T.TrivialAugmentWide()], 'randaug': [T.RandAugment()]}
        if aug not in extras:
            raise ValueError(f'Unknown augmentation: {aug}')
        ops += extras[aug]
    else:
        ops = [T.Resize(round(img_size / .875)), T.CenterCrop(img_size)]
    return T.Compose(ops + [T.ToTensor(), T.Normalize(mean, std)])

class DeepWeedsDataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir, self.transform = Path(images_dir), transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        with Image.open(self.images_dir / row.Filename) as raw:
            image = raw.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, int(row.Label), row.Filename

def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)

def make_loader(df, images_dir, transform, batch_size, train, sampler=None, num_workers=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    if sampler not in (None, 'balanced'):
        raise ValueError('sampler must be None or balanced')
    chosen = None
    if train and sampler == 'balanced':
        counts = df.Label.value_counts()
        weights = [1. / counts[int(y)] for y in df.Label]
        chosen = WeightedRandomSampler(weights, len(df), replacement=True, generator=generator)
    return DataLoader(DeepWeedsDataset(df, images_dir, transform), batch_size=batch_size,
                      shuffle=train and chosen is None, sampler=chosen,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=train and len(df) >= batch_size,
                      worker_init_fn=seed_worker, generator=generator)
