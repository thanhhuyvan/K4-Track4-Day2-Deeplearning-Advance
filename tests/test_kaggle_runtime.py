"""Real CPU checks; synthetic data here is only a test fixture, never lab results."""
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from PIL import Image
from kaggle.code import losses, inference, model as models, dataset, suite
from kaggle.code.train import Config, run, run_dir, EMA

torch.set_num_threads(1)

class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Conv2d(3, 6, 3, padding=1), torch.nn.BatchNorm2d(6),
                                        torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
        self.head = torch.nn.Linear(6, 9)
        self.pretrained_cfg = {'mean':dataset.IMAGENET_MEAN, 'std':dataset.IMAGENET_STD, 'tag':'test-fixture'}
    def forward(self, x): return self.head(self.body(x))
    def get_classifier(self): return self.head

def tiny_build(*args, **kwargs):
    model = TinyModel()
    if kwargs.get('init') == 'frozen': models.freeze_backbone(model)
    return model

class NumericalTests(unittest.TestCase):
    def test_split_without_species_reaches_integrity_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = []
            for i in range(3):
                name = f'image{i}.jpg'; (Path(tmp)/name).touch()
                frames.append(pd.DataFrame({'Filename':[name], 'Label':[i]}))
            # Small fixture must fail the size check, not the optional metadata check.
            with self.assertRaisesRegex(ValueError, 'exactly 17509'):
                dataset.check_split(*frames, tmp)
            invalid = frames[0].rename(columns={'Label':'wrong'})
            with self.assertRaisesRegex(ValueError, 'actual columns'):
                dataset.check_split(invalid, *frames[1:], tmp)

    def test_focal_and_smoothing_reduce_to_ce(self):
        torch.manual_seed(10)
        x, y = torch.randn(12, 9), torch.arange(12) % 9
        ce = torch.nn.functional.cross_entropy(x, y)
        self.assertLess(float((losses.FocalLoss(0)(x, y)-ce).abs()), 1e-6)
        self.assertLess(float((losses.LabelSmoothingCE(0)(x, y)-ce).abs()), 1e-6)

    def test_cutmix_area_matches_label_weight(self):
        x = torch.stack([torch.zeros(3, 8, 8), torch.ones(3, 8, 8)])
        y = torch.tensor([0,1])
        with patch('numpy.random.beta', return_value=.3), patch('torch.randperm', return_value=torch.tensor([1,0])):
            mixed, (a,b,lam) = losses.mix_batch(x,y)
        self.assertAlmostEqual(float(mixed[0].mean()), 1-lam)
        self.assertTrue(torch.equal(b, y.flip(0)))
        self.assertTrue(torch.equal(x[0], torch.zeros_like(x[0])))

    def test_bn_fusion_and_freezing(self):
        model = TinyModel().eval(); x = torch.randn(3,3,16,16)
        fused = inference.fuse_conv_bn(model)
        self.assertLess(float((model(x)-fused(x)).abs().max().detach()), 1e-5)
        models.freeze_backbone(model)
        self.assertFalse(any(p.requires_grad for p in model.body.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.head.parameters()))
        groups = models.param_groups(model, 1e-4, 1e-3, .05)
        ids = [id(p) for g in groups for p in g['params']]
        self.assertEqual(len(ids), len(set(ids)))
        bias_group = next(g for g in groups if any(p is model.head.bias for p in g['params']))
        self.assertEqual(bias_group['weight_decay'], 0)

    def test_temperature_improves_nll_preserves_argmax(self):
        rng = np.random.default_rng(5)
        z = rng.normal(size=(80,9))*8; y = rng.integers(0,9,80)
        t = inference.fit_temperature(z,y)
        before, after = inference.softmax(z), inference.apply_temperature(z,t)
        self.assertTrue(np.array_equal(before.argmax(1),after.argmax(1)))
        self.assertLessEqual(-np.log(after[np.arange(80), y]).mean(), -np.log(before[np.arange(80),y]).mean()+1e-6)

    def test_ema_copies_bn_buffers(self):
        model = TinyModel(); ema = EMA(model,.9)
        with torch.no_grad(): model.body[1].running_mean.fill_(7.)
        ema.update(model)
        self.assertTrue(torch.equal(ema.model.body[1].running_mean, model.body[1].running_mean))

class WorkflowTests(unittest.TestCase):
    def test_train_screen_final_export_and_rerun_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); images=root/'images'; labels=root/'labels'; images.mkdir(); labels.mkdir()
            all_rows=[]
            for part, repeats in [('train',2),('val',1),('test',1)]:
                rows=[]
                for label in range(9):
                    for i in range(repeats):
                        name=f'{part}_{label}_{i}.png'
                        rgb=np.zeros((32,32,3), dtype=np.uint8); rgb[:]=[(label*25)%255,(label*40)%255,(label*60)%255]
                        Image.fromarray(rgb).save(images/name)
                        rows.append(dict(Filename=name,Label=label,Species=dataset.CLASS_NAMES[label]))
                pd.DataFrame(rows).to_csv(labels/f'{part}_subset0.csv',index=False); all_rows+=rows
            pd.DataFrame(all_rows).to_csv(labels/'labels.csv',index=False)
            out=root/'output'; out.mkdir()
            cfg=Config(exp_id='SMOKE',backbone='fixture1',images_dir=str(images),labels_dir=str(labels),epochs=1,batch_size=9,img_size=32,
                       pretrained=False,num_workers=0,out_dir=str(out/'runs'),pred_dir=str(out/'predictions'),
                       curves_dir=str(out/'curves'))
            # Only size/count checks and model size are mocked. Real loaders, losses, train,
            # checkpoint IO, inference, metrics, official CLI and output export all run.
            with patch.object(dataset,'check_split',return_value={'fixture':True}), \
                 patch.object(models,'build_model',side_effect=tiny_build), \
                 patch.object(models,'count_gmacs',return_value=.001):
                result=run(cfg); self.assertTrue((run_dir(cfg)/'best.pt').exists())
                with patch('kaggle.code.train.train_one_epoch', side_effect=AssertionError('must reuse completed run')):
                    self.assertEqual(run(cfg)['macro_f1'],result['macro_f1'])
                with self.assertRaises(ValueError): run(replace(cfg,epochs=2))
                # Simulate interrupted post-training export: last checkpoint recovers without retraining.
                (run_dir(cfg)/'result.json').unlink()
                with patch('kaggle.code.train.train_one_epoch', side_effect=AssertionError('must resume completed epochs')):
                    run(cfg)
                plan=suite.screen(cfg,out,backbones=['fixture1','fixture2','fixture3','fixture4','fixture5'])
                rows=suite.finalize(out,(0,1,2)); self.assertEqual(len(rows),6)
                with patch.object(suite,'predict_method', side_effect=AssertionError('must never rerun test')):
                    self.assertEqual(len(suite.finalize(out,(0,1,2))),6)
                with self.assertRaises(RuntimeError): suite.screen(cfg,out)
                suite.export(out)
                suite.export(out)
                self.assertFalse((out/'submission'/'logs'/'submission').exists())
                self.assertTrue((out/'results.xlsx').exists())
                self.assertTrue((out/'deepweeds_submission.zip').exists())
                self.assertEqual(len(list((out/'predictions').glob('*_test.csv'))),6)
                with pd.ExcelFile(out/'results.xlsx') as workbook:
                    self.assertTrue({'Backbones','Training','Inference','Final','PerClass','Latency','Summary'}.issubset(workbook.sheet_names))

if __name__ == '__main__': unittest.main()
