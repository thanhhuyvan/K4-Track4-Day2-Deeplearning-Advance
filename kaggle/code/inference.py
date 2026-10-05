import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def softmax(logits):
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    x = np.exp(x)
    return x / x.sum(axis=-1, keepdims=True)

def predict_logits(model, loader, device, view=None):
    model.eval()
    filenames, labels, logits = [], [], []
    with torch.inference_mode():
        for x, y, names in loader:
            x = x.to(device, non_blocking=True)
            output = model(x if view is None else view(x))
            filenames.extend(names)
            labels.extend(y.tolist())
            logits.append(output.float().cpu().numpy())
    return filenames, np.asarray(labels), np.concatenate(logits)

def view_identity(x): return x
def view_hflip(x): return x.flip(-1)

def views_multicrop(x, crop):
    h, w = x.shape[-2:]
    if crop > min(h, w): raise ValueError('crop exceeds image dimensions')
    return [x[..., y:y+crop, z:z+crop] for y, z in
            [(0, 0), (0, w-crop), (h-crop, 0), (h-crop, w-crop), ((h-crop)//2, (w-crop)//2)]]

def views_multiscale(x, sizes):
    return [F.interpolate(x, size=(s, s), mode='bilinear', align_corners=False) for s in sizes]

def aggregate_views(logits_per_view, space='prob'):
    if space == 'prob': return np.mean([softmax(x) for x in logits_per_view], axis=0)
    if space == 'logit': return softmax(np.mean(logits_per_view, axis=0))
    raise ValueError(space)

def ensemble_probs(list_of_probs):
    if not list_of_probs or len({p.shape for p in list_of_probs}) != 1:
        raise ValueError('Aligned, equally sized probability arrays required')
    p = np.mean(list_of_probs, axis=0)
    return p / p.sum(1, keepdims=True)

def fit_temperature(val_logits, val_labels):
    logits = torch.as_tensor(val_logits, dtype=torch.float64)
    labels = torch.as_tensor(val_labels, dtype=torch.long)
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=.1, max_iter=80, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.clamp(-4, 4).exp(), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.detach().clamp(-4, 4).exp())

def apply_temperature(logits, T):
    if not np.isfinite(T) or T <= 0: raise ValueError('T must be positive')
    return softmax(np.asarray(logits) / T)

def fuse_conv_bn(model):
    # Only fuse graph relationships known to be safe, not arbitrary registered siblings.
    result = copy.deepcopy(model).eval()
    def walk(module):
        for child in list(module.children()): walk(child)
        if isinstance(module, nn.Sequential):
            keys = list(module._modules)
            for a, b in zip(keys, keys[1:]):
                if isinstance(module._modules[a], nn.Conv2d) and isinstance(module._modules[b], nn.BatchNorm2d):
                    module._modules[a] = torch.nn.utils.fuse_conv_bn_eval(module._modules[a], module._modules[b])
                    module._modules[b] = nn.Identity()
        # timm ResNet blocks and stem: these convN -> bnN pairs are explicit in forward.
        if module.__class__.__module__.startswith('timm.models.resnet'):
            for i in (1, 2, 3):
                conv, bn = getattr(module, f'conv{i}', None), getattr(module, f'bn{i}', None)
                if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                    setattr(module, f'conv{i}', torch.nn.utils.fuse_conv_bn_eval(conv, bn))
                    setattr(module, f'bn{i}', nn.Identity())
    walk(result)
    return result
