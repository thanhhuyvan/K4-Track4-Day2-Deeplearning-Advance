import argparse
import copy
import json
import importlib.metadata
import math
import os
import random
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from torch.nn import functional as F
from . import dataset, model as models, losses
from .inference import softmax
from eval import compute_metrics, save_predictions

@dataclass
class Config:
    exp_id: str = 'T00'
    seed: int = 0
    fold: int = 0
    backbone: str = 'resnet50'
    init: str = 'finetune'
    drop_rate: float = 0.
    img_size: int = 224
    aug: str = 'basic'
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.
    loss: str = 'ce'
    label_smoothing: float = 0.
    focal_gamma: float = 2.
    class_weight_beta: float | None = None
    epochs: int = 12
    batch_size: int = 32
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = .05
    warmup_epochs: float = 1.
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    images_dir: str = 'data/images'
    labels_dir: str = 'data/labels'
    out_dir: str = 'runs'
    pred_dir: str = 'predictions'
    curves_dir: str = 'curves'
    save_test_predictions: bool = False
    pretrained: bool = True
    optimizer: str = 'adamw'

def run_dir(cfg): return Path(cfg.out_dir) / cfg.exp_id / f'seed{cfg.seed}'
def pred_path(cfg, split): return Path(cfg.pred_dir) / f'{cfg.exp_id}_seed{cfg.seed}_{split}.csv'

def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    tmp.replace(path)

def atomic_save(obj, path):
    tmp = Path(str(path) + '.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_optimizer(model, cfg):
    groups = models.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    if cfg.optimizer == 'adamw': return torch.optim.AdamW(groups)
    if cfg.optimizer == 'sgd': return torch.optim.SGD(groups, momentum=.9)
    raise ValueError(cfg.optimizer)

def build_scheduler(optimizer, cfg, steps_per_epoch):
    total, warmup = cfg.epochs * steps_per_epoch, int(cfg.warmup_epochs * steps_per_epoch)
    def factor(step):
        if step < warmup: return (step + 1) / max(1, warmup)
        progress = min(1., (step - warmup) / max(1, total - warmup))
        return .5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

class EMA:
    def __init__(self, model, decay):
        self.model, self.decay = copy.deepcopy(model).eval(), decay
        self.model.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        source = model.state_dict()
        parameter_names = set(dict(model.named_parameters()))
        for key, value in self.model.state_dict().items():
            if key in parameter_names and value.is_floating_point():
                value.mul_(self.decay).add_(source[key], alpha=1-self.decay)
            else:
                # Copy BN running statistics/counters instead of averaging their history twice.
                value.copy_(source[key])

def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg, device, ema=None):
    model.train()
    if cfg.init == 'frozen':
        model.eval(); model.get_classifier().train()
    total, n, correct = 0., 0, 0
    progress = tqdm(loader, desc=f'{cfg.exp_id} seed={cfg.seed} train', leave=False, mininterval=1.)
    for x, y, _ in progress:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        targets = y
        if cfg.mix: x, targets = losses.mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == 'cuda'):
            logits = model(x)
            loss = losses.mixed_loss(criterion, logits, targets) if cfg.mix else criterion(logits, y)
        if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer); scaler.update()
        if scaler.get_scale() >= previous_scale:
            scheduler.step()
            if ema: ema.update(model)
        total += loss.item() * len(y); n += len(y)
        postfix = {'loss':f'{total/n:.4f}', 'lr':f'{optimizer.param_groups[0]["lr"]:.2e}'}
        if not cfg.mix:
            correct += int((logits.detach().argmax(1) == y).sum())
            postfix['acc'] = f'{correct/n:.2%}'
        progress.set_postfix(postfix, refresh=False)
    return {'train_loss': total / n, 'train_acc': correct/n if not cfg.mix else None,
            'lr': optimizer.param_groups[0]['lr']}

def evaluate(model, loader, criterion, device):
    model.eval()
    names, labels, logits, total = [], [], [], 0.
    with torch.inference_mode():
        for x, y, files in tqdm(loader, desc='Validation', leave=False, mininterval=1.):
            x, y = x.to(device), y.to(device)
            z = model(x)
            total += criterion(z, y).item() * len(y)
            names.extend(files); labels.extend(y.cpu().tolist()); logits.append(z.float().cpu().numpy())
    return names, np.asarray(labels), np.concatenate(logits), total / len(labels)

def plot_curves(history, path, title):
    import matplotlib.pyplot as plt
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(df.epoch, df.train_loss, label='train objective')
    axes[0].plot(df.epoch, df.val_loss, label='val CE')
    axes[0].set_ylabel('Loss'); axes[0].legend()
    axes[1].plot(df.epoch, df.macro_f1); axes[1].set_ylabel('Macro-F1 val')
    axes[2].plot(df.epoch, df.lr); axes[2].set_ylabel('Learning rate')
    for ax in axes: ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
    fig.suptitle(title); fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160); plt.close(fig)

def run(cfg):
    if cfg.save_test_predictions:
        raise ValueError('Use suite.finalize() for locked, exactly-once test evaluation')
    if cfg.epochs < 1 or cfg.batch_size < 2: raise ValueError('epochs >= 1 and batch_size >= 2 required')
    set_seed(cfg.seed)
    folder = run_dir(cfg); folder.mkdir(parents=True, exist_ok=True)
    config_file = folder / 'config.json'
    if config_file.exists() and json.loads(config_file.read_text()) != asdict(cfg):
        raise ValueError(f'{folder}: configuration changed; use another exp_id')
    if (folder / 'result.json').exists():
        cached = json.loads((folder / 'result.json').read_text())
        print(f'{cfg.exp_id} seed={cfg.seed}: loaded completed run | '
              f'val_acc={cached["top1"]:.2%} | val_F1={cached["macro_f1"]:.4f}', flush=True)
        return cached
    write_json(config_file, asdict(cfg))
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
    write_json(folder / 'split_checks.json', dataset.check_split(train_df, val_df, test_df, cfg.images_dir))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Do not redownload pretrained weights when resuming a saved run.
    model = models.build_model(cfg.backbone, cfg.pretrained and not (folder/'last.pt').exists(),
                               drop_rate=cfg.drop_rate, init=cfg.init).to(device)
    pretrained_cfg = model.pretrained_cfg
    mean, std = pretrained_cfg.get('mean', dataset.IMAGENET_MEAN), pretrained_cfg.get('std', dataset.IMAGENET_STD)
    train_transform = dataset.build_transforms(True, cfg.img_size, cfg.aug, mean, std)
    val_transform = dataset.build_transforms(False, cfg.img_size, mean=mean, std=std)
    val_loader = dataset.make_loader(val_df, cfg.images_dir, val_transform, cfg.batch_size, False,
                                     num_workers=cfg.num_workers, seed=cfg.seed)
    steps = len(train_df) // cfg.batch_size
    criterion = losses.build_criterion(cfg.loss, smoothing=cfg.label_smoothing or .1,
        gamma=cfg.focal_gamma, weight=losses.class_weights(
            train_df.Label.value_counts().reindex(range(9), fill_value=0).values,
            cfg.class_weight_beta or 0.)).to(device)
    val_criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp and device.type == 'cuda')
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None
    start, history, best, best_epoch = 0, [], -1., 0
    metadata = {'tag': pretrained_cfg.get('tag', ''), 'pretrained_cfg': pretrained_cfg,
                'params_m': models.count_params(model), 'gmac': models.count_gmacs(model, cfg.img_size),
                'gmac_tool': 'fvcore; one multiply-add counts as one; unsupported ops printed',
                'torch': torch.__version__, 'timm': __import__('timm').__version__,
                'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU',
                'normalization': {'mean': mean, 'std': std}}
    metadata['versions'] = {name: importlib.metadata.version(name) for name in
                            ('torch', 'torchvision', 'timm', 'numpy', 'pandas', 'matplotlib', 'fvcore', 'openpyxl')}
    write_json(folder / 'environment.json', metadata)
    if (folder/'last.pt').exists():
        checkpoint = torch.load(folder/'last.pt', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler']); scaler.load_state_dict(checkpoint['scaler'])
        if ema: ema.model.load_state_dict(checkpoint['ema'])
        start, history, best, best_epoch = checkpoint['epoch'], checkpoint['history'], checkpoint['best'], checkpoint['best_epoch']
    for epoch in range(start, cfg.epochs):
        print(f'\n{cfg.exp_id} | {cfg.backbone} | seed={cfg.seed} | epoch {epoch+1}/{cfg.epochs}', flush=True)
        # Epoch-specific seed makes restarted epoch reproduce data order and augmentation.
        set_seed(cfg.seed * 100000 + epoch)
        loader = dataset.make_loader(train_df, cfg.images_dir, train_transform, cfg.batch_size, True,
                                     cfg.sampler, cfg.num_workers, cfg.seed * 100000 + epoch)
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()
        stats = train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        if device.type == 'cuda': torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        chosen = ema.model if ema else model
        names, y, z, val_loss = evaluate(chosen, val_loader, val_criterion, device)
        metrics = compute_metrics(y, z.argmax(1), softmax(z))
        stats.update(epoch=epoch+1, val_loss=val_loss, macro_f1=metrics['macro_f1'],
                     top1=metrics['top1'], train_seconds=elapsed)
        history.append(stats)
        if metrics['macro_f1'] > best:
            best, best_epoch = metrics['macro_f1'], epoch+1
            atomic_save({'model': chosen.state_dict(), 'epoch': best_epoch}, folder/'best.pt')
        atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                         scaler=scaler.state_dict(), ema=ema.model.state_dict() if ema else None,
                         epoch=epoch+1, history=history, best=best, best_epoch=best_epoch), folder/'last.pt')
        pd.DataFrame(history).to_csv(folder/'history.csv', index=False)
        plot_curves(history, Path(cfg.curves_dir)/f'{cfg.exp_id}_seed{cfg.seed}_{cfg.backbone}.png',
                    f'{cfg.exp_id} seed={cfg.seed} {cfg.backbone}')
        train_acc = f'{stats["train_acc"]:.2%}' if stats['train_acc'] is not None else 'N/A (mixed labels)'
        print(f'{cfg.exp_id} epoch={epoch+1}/{cfg.epochs} | '
              f'train_loss={stats["train_loss"]:.4f} | train_acc={train_acc} | '
              f'val_loss={val_loss:.4f} | val_acc={metrics["top1"]:.2%} | '
              f'val_F1={metrics["macro_f1"]:.4f} | ECE={metrics["ece"]:.4f} | '
              f'lr={stats["lr"]:.2e} | train_time={elapsed:.1f}s | '
              f'best_F1={best:.4f} (epoch {best_epoch})', flush=True)
    model.load_state_dict(torch.load(folder/'best.pt', map_location=device, weights_only=False)['model'])
    names, y, z, _ = evaluate(model, val_loader, val_criterion, device)
    np.savez_compressed(folder/'val_logits.npz', filenames=np.asarray(names), y=y, logits=z)
    save_predictions(pred_path(cfg, 'val'), names, y, softmax(z))
    metrics = compute_metrics(y, z.argmax(1), softmax(z))
    result = {**asdict(cfg), 'best_epoch': best_epoch, 'macro_f1': metrics['macro_f1'],
              'top1': metrics['top1'], 'ece': metrics['ece'], 'params_m': metadata['params_m'],
              'gmac': metadata['gmac'], 'tag': metadata['tag'],
              'train_seconds': float(np.mean([r['train_seconds'] for r in history]))}
    write_json(folder/'result.json', result)
    del model, optimizer, ema
    if device.type == 'cuda': torch.cuda.empty_cache()
    return result

def parse_overrides(pairs):
    defaults = asdict(Config()); overrides = {}
    for pair in pairs:
        key, value = pair.split('=', 1)
        if key not in defaults: raise ValueError(f'Unknown config key: {key}')
        if value.lower() in ('none', 'null'): parsed = None
        elif key in ('amp', 'pretrained', 'save_test_predictions'):
            if value.lower() not in ('true', 'false'): raise ValueError('Boolean must be true/false')
            parsed = value.lower() == 'true'
        elif key in ('ema_decay', 'class_weight_beta'): parsed = float(value)
        elif isinstance(defaults[key], int): parsed = int(value)
        elif isinstance(defaults[key], float): parsed = float(value)
        else: parsed = value
        overrides[key] = parsed
    return overrides

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--set', nargs='*', default=[])
    print(run(Config(**parse_overrides(parser.parse_args().set))))

if __name__ == '__main__': main()
