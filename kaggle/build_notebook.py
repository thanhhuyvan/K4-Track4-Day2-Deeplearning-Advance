"""Rebuild the portable notebook with embedded, versioned source files."""
import ast
import base64
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def markdown(text): return {'cell_type':'markdown', 'metadata':{}, 'source':text.splitlines(True)}
def code(text): return {'cell_type':'code', 'execution_count':None, 'metadata':{}, 'outputs':[], 'source':text.splitlines(True)}

def build():
    payload = {'eval.py': (ROOT.parent/'eval.py').read_text(encoding='utf-8')}
    for file in sorted((ROOT/'code').glob('*.py')):
        ast.parse(file.read_text(encoding='utf-8'))
        payload['lab_code/'+file.name] = file.read_text(encoding='utf-8')
    encoded = base64.b64encode(json.dumps(payload, ensure_ascii=False).encode()).decode()
    cells = [markdown('''# DeepWeeds — bản v3: hiển thị số liệu training

Notebook độc lập: chứa toàn bộ source, không cần clone GitHub hoặc upload từng script.
**Bản v3:** code training nằm trong ô riêng, hiển thị batch progress, train/val loss,
accuracy, macro-F1 hiện tại và tốt nhất, ECE, LR, thời gian mỗi epoch.
Notebook mới chưa chạy nên không có kết quả lưu sẵn; số liệu xuất hiện khi chạy ô training.

1. Settings → Accelerator → GPU; Internet ON để cài thư viện/tải trọng số ImageNet.
2. Add Input: dataset DeepWeeds có `images/`, `labels/`, CSV gốc fold 0 và `labels.csv`.
3. Chạy lần lượt các ô. Mặc định chạy đầy đủ 12 epoch, baseline + final với 3 seed.
4. Kết quả ở `/kaggle/working/deepweeds_lab/`; ZIP bài nộp không chứa dataset/checkpoint.

**Chạy đầy đủ mất nhiều giờ GPU.** Có thể chỉ chạy từng giai đoạn bằng RUN_SCREEN/RUN_FINAL.
Không đổi epoch/config giữa các lần tiếp tục cùng thư mục output. Không chỉnh cấu hình sau khi mở kết quả test.
Số liệu chỉ được tạo từ lần chạy thật; bản report tự động cần bổ sung nhận xét của bạn.
Nguồn hướng dẫn Kaggle: https://www.kaggle.com/docs/notebooks
API timm: https://huggingface.co/docs/timm/reference/models
'''), code('''import importlib.util, subprocess, sys
required = {"timm":"timm>=1.0,<1.1", "fvcore":"fvcore", "openpyxl":"openpyxl",
            "tabulate":"tabulate", "pandas":"pandas>=2.1,<3", "matplotlib":"matplotlib",
            "torchvision":"torchvision"}
missing = [spec for module, spec in required.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
import torch, pandas as pd, timm
print("torch", torch.__version__, "timm", timm.__version__, "pandas", pd.__version__)
if not torch.cuda.is_available():
    raise RuntimeError("Bật GPU trong Settings trước khi chạy bài đầy đủ.")
print("GPU:", torch.cuda.get_device_name(0))
'''), markdown('## 1. Giải nén code nhúng trong notebook'), code(f'''import base64, json, os, sys
from pathlib import Path
WORKSPACE = Path("/kaggle/working/deepweeds_project")
WORKSPACE.mkdir(parents=True, exist_ok=True)
SOURCES = json.loads(base64.b64decode("{encoded}").decode("utf-8"))
for name, source in SOURCES.items():
    target = WORKSPACE / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
os.chdir(WORKSPACE)
sys.path.insert(0, str(WORKSPACE))
for module_name in list(sys.modules):
    if module_name == "lab_code" or module_name.startswith("lab_code."):
        del sys.modules[module_name]
print("Notebook version: v3 — live training metrics")
print("Đã tạo", len(SOURCES), "source files tại", WORKSPACE)
'''), markdown('''## 2. Cấu hình chạy

`EPOCHS=12` là bài đầy đủ. Giảm xuống 1 chỉ để kiểm tra chạy được, không dùng làm kết quả bài lab.
Nếu notebook tìm nhiều bộ dữ liệu, nhập đường dẫn IMAGES_DIR và LABELS_DIR.
Muốn tiếp tục phiên mới: Add Input output của phiên trước và đặt RESTORE_FROM đến thư mục `deepweeds_lab` đó.
Chỉ đặt RUN_SCREEN=False khi đã có `selection.json`; RUN_FINAL=False để dừng trước test.
'''), code('''from lab_code.train import Config
from lab_code import dataset, suite
import shutil
import inspect
from lab_code import train
assert "tqdm" in inspect.getsource(train.train_one_epoch)
assert "val_F1=" in inspect.getsource(train.run)
print("v3 đã nạp: batch progress + train/val loss + accuracy + macro-F1 + ECE + LR + thời gian")

EPOCHS = 12
BATCH_SIZE = 32  # dùng cùng giá trị cho tất cả backbone; giảm trước khi bắt đầu nếu OOM
NUM_WORKERS = 2
IMAGES_DIR = None
LABELS_DIR = None
OUTPUT = Path("/kaggle/working/deepweeds_lab")
RESTORE_FROM = None  # ví dụ "/kaggle/input/previous-run/deepweeds_lab"
RUN_SCREEN = True
RUN_FINAL = True
SEEDS = (0, 1, 2)

if RESTORE_FROM:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError("OUTPUT đã có dữ liệu; không ghi đè bằng output phiên trước.")
    shutil.copytree(RESTORE_FROM, OUTPUT, dirs_exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
IMAGES_DIR, LABELS_DIR = dataset.discover("/kaggle/input", IMAGES_DIR, LABELS_DIR)
base = Config(images_dir=IMAGES_DIR, labels_dir=LABELS_DIR,
              epochs=EPOCHS, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
              out_dir=str(OUTPUT/"runs"), pred_dir=str(OUTPUT/"predictions"),
              curves_dir=str(OUTPUT/"curves"))
print("Images:", IMAGES_DIR, "Labels:", LABELS_DIR, "Output:", OUTPUT)
'''), markdown('## 3. EDA và kiểm tra pipeline trước khi chạy thí nghiệm'), code('''from IPython.display import display, Image, FileLink
suite.eda(base, OUTPUT/"eda")
suite.check_pipeline(base, OUTPUT/"eda")
display(Image(filename=str(OUTPUT/"eda"/"class_distribution.png")))
display(Image(filename=str(OUTPUT/"eda"/"augmentations.png")))
display(Image(filename=str(OUTPUT/"eda"/"cutmix.png")))
'''), markdown('''## 4. Sàng lọc backbone, training, inference trên validation

5 backbone: ResNet50, ResNeXt50, ConvNeXt-T, DeiT-S, EfficientNet-B0.
Training: scratch/frozen/finetune; CutMix/color; CE/smoothing/focal/weighted CE; kết hợp tốt nhất.
Inference: 1-view, hflip probability/logit, five-crop, temperature scaling, AMP.
Lưu mỗi epoch; chạy lại cùng config để tiếp tục. Macro-F1 chọn checkpoint; hòa chọn epoch sớm hơn.
'''), code('''if RUN_SCREEN:
    selection = suite.screen(base, OUTPUT)
else:
    selection = json.loads((OUTPUT/"selection.json").read_text())
print("Selected training:", selection["chosen"])
print("Selected inference:", selection["method"])
for title, rows in [("Backbones", selection["backbone_rows"]), ("Training", selection["training_rows"])]:
    print(title)
    display(pd.DataFrame(rows)[["exp_id", "backbone", "macro_f1", "top1", "best_epoch", "train_seconds"]])
'''), markdown('''## 5. Chung kết: khóa cấu hình và chạy test

Sau bước này không quay lại điều chỉnh dựa trên test. Baseline T00 + single, final F01 + phương pháp đã chọn;
mỗi nhóm 3 seed. Final luôn fit temperature trên val theo quy tắc đã khóa, rồi áp dụng test.
Test chạy đúng một lần mỗi seed/nhóm; kết quả hoàn thành được tái sử dụng khi chạy lại.
Nếu phiên ngắt giữa test, script dừng và giữ dấu vết thay vì âm thầm chạy test lại.
'''), code('''if RUN_FINAL:
    final_rows = suite.finalize(OUTPUT, SEEDS)
    display(pd.DataFrame(final_rows))
else:
    print("Chưa chạy test. Chỉ có kết quả validation.")
'''), markdown('''## 6. Xuất Excel, báo cáo, biểu đồ và ZIP bài nộp

Excel có Backbones, Training, Inference, Final, PerClass, Latency, Summary và FinalSummary.
Thêm MSSV, họ tên, link notebook và phân tích của bạn trước khi nộp; không tự điền số chưa chạy.
'''), code('''suite.export(OUTPUT)
display(Image(filename=str(OUTPUT/"curves"/"inference_tradeoff.png")))
for name in ("results.xlsx", "report.md", "deepweeds_submission.zip"):
    display(FileLink(str(OUTPUT/name)))
print("Tải các file từ Output của notebook sau Save Version.")
''')]
    cells[4:4] = [markdown('''### Code training có thể xem và sửa trực tiếp

Ô dưới ghi `train.py` vào workspace Kaggle. Bạn **không cần upload file .py**.
Tìm `train_one_epoch` để xem batch progress, hoặc `train_loss=` để xem số liệu sau từng epoch.
Chạy ô này trước ô cấu hình. Nếu đã import code trong phiên cũ, khởi động lại session rồi Run All.
'''), code('%%writefile /kaggle/working/deepweeds_project/lab_code/train.py\n' + payload['lab_code/train.py'])]
    notebook = dict(cells=cells, metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                    'language_info':{'name':'python','version':'3.11'},
                    'kaggle':{'isInternetEnabled':True,'language':'python','sourceType':'notebook','isGpuEnabled':True}},
                    nbformat=4, nbformat_minor=5)
    for i, cell in enumerate(cells): cell['id'] = f'cell-{i:02d}'
    path = ROOT/'deepweeds_kaggle.ipynb'
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    (ROOT/'deepweeds_kaggle_v3.ipynb').write_text(path.read_text(encoding='utf-8'), encoding='utf-8')
    print(path, path.stat().st_size, 'bytes')
    with zipfile.ZipFile(ROOT/'deepweeds_kaggle_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for file in [path, ROOT/'README.md', ROOT/'requirements.txt', ROOT/'build_notebook.py']:
            archive.write(file, file.name)
        for file in sorted((ROOT/'code').glob('*.py')):
            archive.write(file, 'lab_code/'+file.name)
        archive.write(ROOT.parent/'eval.py', 'eval.py')

if __name__ == '__main__': build()
