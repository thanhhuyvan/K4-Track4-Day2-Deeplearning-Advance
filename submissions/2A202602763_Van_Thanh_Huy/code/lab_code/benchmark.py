import time
import numpy as np
import torch

def bench(fn, warmup=10, iters=50, sync=None):
    if warmup < 10 or iters < 50: raise ValueError('At least 10 warmups and 50 measured iterations required')
    sync = sync or (lambda: None)
    for _ in range(warmup): fn()
    samples = []
    for _ in range(iters):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - start) * 1000)
    return dict(zip(('p50', 'p95', 'p99'), map(float, np.percentile(samples, [50, 95, 99]))),
                mean=float(np.mean(samples)), n=iters)

def latency_report(model, batch_size, img_size, dtype='fp32', device='cuda', warmup=10, iters=50):
    model.eval()
    if dtype not in ('fp32', 'amp', 'fp16'): raise ValueError(dtype)
    if str(device).startswith('cpu') and dtype != 'fp32': raise ValueError('CPU benchmark uses fp32')
    original_dtype = next(model.parameters()).dtype
    try:
        model.to(device=device, dtype=torch.float16 if dtype == 'fp16' else torch.float32)
        x = torch.randn(batch_size, 3, img_size, img_size, device=device,
                        dtype=torch.float16 if dtype == 'fp16' else torch.float32)
        with torch.inference_mode(), torch.autocast(device_type=torch.device(device).type, enabled=dtype == 'amp'):
            report = bench(lambda: model(x), warmup, iters,
                           torch.cuda.synchronize if str(device).startswith('cuda') else None)
        report.update(gpu=torch.cuda.get_device_name() if str(device).startswith('cuda') else 'CPU',
                      dtype=dtype, batch=batch_size, img_size=img_size,
                      images_per_s=batch_size * 1000 / report['p50'], torch=torch.__version__,
                      preprocessing='excluded; synthetic tensor on device')
        return report
    finally:
        model.to(dtype=original_dtype)

def tta_latency(model, k_views, **kw):
    # This wrapper measures repeated forwards; use suite benchmark for actual TTA transforms/aggregation.
    class Repeated(torch.nn.Module):
        def __init__(self, base): super().__init__(); self.base = base
        def forward(self, x): return torch.stack([self.base(x) for _ in range(k_views)]).mean(0)
    return latency_report(Repeated(model), **kw)
