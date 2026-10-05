import torch
import timm

SUGGESTED_BACKBONES = ['resnet50', 'resnext50_32x4d', 'convnext_tiny',
                       'deit_small_patch16_224', 'efficientnet_b0']

def build_model(name, pretrained=True, num_classes=9, drop_rate=0., init='finetune'):
    if init not in ('scratch', 'frozen', 'finetune'):
        raise ValueError(init)
    model = timm.create_model(name, pretrained=pretrained and init != 'scratch',
                              num_classes=num_classes, drop_rate=drop_rate)
    if init == 'frozen':
        freeze_backbone(model)
    return model

def freeze_backbone(model):
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.get_classifier().parameters():
        p.requires_grad_(True)
    model.eval()

def param_groups(model, lr_backbone, lr_head, weight_decay):
    head = {id(p) for p in model.get_classifier().parameters()}
    groups = {}
    for p in model.parameters():
        if p.requires_grad:
            # All bias and normalization parameters, including in the head, have no decay.
            key = (lr_head if id(p) in head else lr_backbone, weight_decay if p.ndim > 1 else 0.)
            groups.setdefault(key, []).append(p)
    return [dict(params=ps, lr=lr, weight_decay=wd) for (lr, wd), ps in groups.items()]

def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def count_gmacs(model, img_size=224):
    from fvcore.nn import FlopCountAnalysis
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    # fvcore cannot count fused scaled_dot_product_attention. Trace the equivalent
    # unfused timm attention so QK^T and AV multiply-adds are included.
    attention_flags = [(module, module.fused_attn) for module in model.modules() if hasattr(module, 'fused_attn')]
    for module, _ in attention_flags: module.fused_attn = False
    try:
        with torch.inference_mode():
            analysis = FlopCountAnalysis(model, torch.zeros(1, 3, img_size, img_size, device=device))
            total = analysis.unsupported_ops_warnings(False).uncalled_modules_warnings(False).total()
        unsupported = dict(analysis.unsupported_ops())
        if unsupported:
            print('GMAC estimate: fvcore unsupported operators:', unsupported)
        return total / 1e9
    finally:
        for module, flag in attention_flags: module.fused_attn = flag
        model.train(was_training)
